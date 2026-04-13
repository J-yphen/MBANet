import argparse
import os

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

import cosnet
import student_cosnet
from align_resize import AlignResize
from zerowaste import ZeroWasteDataset
from specwaste import SpectralWasteDataset


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())


def load_any_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'student' in ckpt:
        msg = model.load_state_dict(ckpt['student'], strict=False)
        print(f'Loaded student state_dict from KD checkpoint: {msg}')
        return ckpt
    return load_checkpoint(model, checkpoint_path, map_location='cpu')


def _format_ratio(num, den):
    return 100.0 * float(num) / (float(den) + 1e-6)


def compute_metrics_from_outputs(dataset, outputs):
    num_classes = len(dataset.CLASSES)
    intersection = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    target = np.zeros(num_classes, dtype=np.float64)
    correct = 0.0
    total = 0.0

    gt_iter = dataset.get_gt_seg_maps()
    for pred, gt in zip(outputs, gt_iter):
        pred = np.asarray(pred, dtype=np.int64)
        gt = np.asarray(gt, dtype=np.int64)

        valid = gt != 255
        pred = pred[valid]
        gt = gt[valid]

        correct += (pred == gt).sum()
        total += gt.size

        for cls in range(num_classes):
            pred_c = pred == cls
            gt_c = gt == cls
            intersection[cls] += np.logical_and(pred_c, gt_c).sum()
            union[cls] += np.logical_or(pred_c, gt_c).sum()
            target[cls] += gt_c.sum()

    class_iou = [_format_ratio(intersection[i], union[i]) for i in range(num_classes)]
    class_acc = [_format_ratio(intersection[i], target[i]) for i in range(num_classes)]

    miou = float(np.mean(class_iou))
    macc = float(np.mean(class_acc))
    aacc = _format_ratio(correct, total)
    return {
        'class_iou': class_iou,
        'class_acc': class_acc,
        'miou': miou,
        'macc': macc,
        'aacc': aacc,
    }


def print_classwise_table(dataset, metrics, title='Metrics'):
    print(f'\n{title}')
    print('+----+----------------------+-----------+-------------------+')
    print('| ID | Class                | IoU (%)   | Pixel Acc (%)     |')
    print('+----+----------------------+-----------+-------------------+')
    for i, cls_name in enumerate(dataset.CLASSES):
        print(
            f'| {i:>2} | {cls_name:<20} | '
            f'{metrics["class_iou"][i]:>9.2f} | {metrics["class_acc"][i]:>17.2f} |'
        )
    print('+----+----------------------+-----------+-------------------+')
    print(
        f'mIoU: {metrics["miou"]:.2f}% | '
        f'Avg Pixel Acc (mAcc): {metrics["macc"]:.2f}% | '
        f'Overall Pixel Acc (aAcc): {metrics["aacc"]:.2f}%'
    )


def run_eval_once(cfg, checkpoint_path, args, distributed, tag='Model'):
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    _ = load_any_checkpoint(model, checkpoint_path)
    model.CLASSES = dataset.CLASSES
    model.PALETTE = dataset.PALETTE
    total_params = count_total_params(model)
    encoder_params = count_total_params(model.backbone)

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir, efficient_test, args.opacity)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    rank, _ = get_dist_info()
    result = None
    if rank == 0:
        metrics = compute_metrics_from_outputs(dataset, outputs)
        print_classwise_table(dataset, metrics, title=f'{tag} Classwise Results')
        result = {
            'tag': tag,
            'metrics': metrics,
            'total_params': total_params,
            'encoder_params': encoder_params,
            'outputs': outputs,
            'dataset': dataset,
        }
    return result

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--teacher-config', default=None, help='teacher config file path for comparison')
    parser.add_argument('--teacher-checkpoint', default=None, help='teacher checkpoint for comparison')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    #parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show', default=False, help='show results')
    parser.add_argument(
        '--show-dir', default='', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        default=True,
        #action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir', default='zerowaste_test_logs', 
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    student_result = run_eval_once(cfg, args.checkpoint, args, distributed, tag='Student')

    teacher_result = None
    if args.teacher_config is not None and args.teacher_checkpoint is not None:
        teacher_cfg = mmcv.Config.fromfile(args.teacher_config)
        if args.options is not None:
            teacher_cfg.merge_from_dict(args.options)
        if teacher_cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        teacher_cfg.model.pretrained = None
        teacher_cfg.data.test.test_mode = True
        teacher_result = run_eval_once(
            teacher_cfg,
            args.teacher_checkpoint,
            args,
            distributed,
            tag='Teacher')

    rank, _ = get_dist_info()
    if rank == 0:
        if student_result is None:
            raise RuntimeError('No student result generated on rank 0.')

        outputs = student_result['outputs']
        dataset = student_result['dataset']
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            dataset.evaluate(outputs, args.eval, **kwargs)

        print('\nSummary')
        print('+---------+--------------+---------------------+---------------------+---------------------+')
        print('| Model   | Params (M)   | Encoder Params (M)  | mIoU (%)            | Avg Pixel Acc (%)   |')
        print('+---------+--------------+---------------------+---------------------+---------------------+')
        print(
            f"| Student | {student_result['total_params'] / 1e6:>12.2f} | "
            f"{student_result['encoder_params'] / 1e6:>19.2f} | "
            f"{student_result['metrics']['miou']:>19.2f} | "
            f"{student_result['metrics']['macc']:>19.2f} |"
        )
        if teacher_result is not None:
            print(
                f"| Teacher | {teacher_result['total_params'] / 1e6:>12.2f} | "
                f"{teacher_result['encoder_params'] / 1e6:>19.2f} | "
                f"{teacher_result['metrics']['miou']:>19.2f} | "
                f"{teacher_result['metrics']['macc']:>19.2f} |"
            )
        print('+---------+--------------+---------------------+---------------------+---------------------+')


if __name__ == '__main__':
    main()
