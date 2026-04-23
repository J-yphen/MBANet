import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
#from mmcv.runner import init_dist
from dist_utils import init_dist
from mmengine.config import Config, DictAction
from version_utils import get_git_hash
#from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
#from mmseg.apis import train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger
import cosnet
import models
from align_resize import AlignResize
from zerowaste import ZeroWasteDataset
from specwaste import SpectralWasteDataset
#torch.autograd.set_detect_anomaly(True)


def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        state_dict = checkpoint_obj.get('state_dict', checkpoint_obj)
        if isinstance(state_dict, dict):
            return state_dict
    return {}


def _collect_checkpoint_load_stats(model, state_dict, backbone_only=False):
    model_state = model.state_dict()
    model_keys = set(model_state.keys())

    if backbone_only:
        target_keys = {k for k in model_keys if k.startswith('backbone.')}
    else:
        target_keys = model_keys

    matched = set()
    shape_mismatch = []
    unexpected = []

    for key, value in state_dict.items():
        mapped_key = key
        if backbone_only and not key.startswith('backbone.'):
            candidate = f'backbone.{key}'
            if candidate in model_keys:
                mapped_key = candidate

        if mapped_key in target_keys:
            model_tensor = model_state[mapped_key]
            if hasattr(value, 'shape') and tuple(value.shape) == tuple(model_tensor.shape):
                matched.add(mapped_key)
            else:
                shape_mismatch.append((mapped_key, tuple(getattr(value, 'shape', ())), tuple(model_tensor.shape)))
        else:
            unexpected.append(mapped_key)

    missing = sorted(list(target_keys - matched))
    return {
        'matched': len(matched),
        'total_target': len(target_keys),
        'missing': missing,
        'shape_mismatch': shape_mismatch,
        'unexpected': unexpected,
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', default=None, help='train config file path')
    parser.add_argument('--work-dir', default='zerowaste_logs', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--no-print-model',
        action='store_true',
        help='do not print the full model architecture in logs')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    load_from_meta = None

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        checkpoint = torch.load(args.load_from, map_location='cpu')
        state_dict = _extract_state_dict(checkpoint)

        keys = list(state_dict.keys())
        has_backbone_keys = any(k.startswith('backbone.') for k in keys)
        has_seg_head_keys = any(k.startswith('decode_head.') or k.startswith('auxiliary_head.') for k in keys)

        load_from_meta = {
            'path': args.load_from,
            'state_dict': state_dict,
            'is_backbone_only': has_backbone_keys and not has_seg_head_keys,
        }

        # Backbone-only checkpoints should initialize cfg.model.pretrained,
        # not cfg.load_from (which expects full segmentor checkpoint keys).
        if has_backbone_keys and not has_seg_head_keys:
            cfg.model.pretrained = args.load_from
            warnings.warn(
                'Detected backbone-only checkpoint in --load-from; '
                'using it as model.pretrained and skipping full-model load.'
            )
        else:
            backbone_pretrained = cfg.model.get('pretrained', None) if cfg.get('model', None) else None
            if backbone_pretrained is not None and osp.abspath(args.load_from) == osp.abspath(backbone_pretrained):
                warnings.warn(
                    'Ignoring --load-from because it matches model.pretrained; '
                    'the file is treated as backbone initialization, not a full segmentation checkpoint.'
                )
            else:
                cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    cfg.device = 'cuda'

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    if not args.no_print_model:
        logger.info(model)
    else:
        logger.info('Model architecture printing disabled by --no-print-model')

    if load_from_meta is not None:
        stats = _collect_checkpoint_load_stats(
            model,
            load_from_meta['state_dict'],
            backbone_only=load_from_meta['is_backbone_only'])
        logger.info(
            'Checkpoint loaded from %s (%d/%d target layers matched).',
            load_from_meta['path'],
            stats['matched'],
            stats['total_target'])

        if stats['shape_mismatch']:
            logger.warning('Found %d shape-mismatched layers while loading %s.',
                           len(stats['shape_mismatch']), load_from_meta['path'])
            for key, ckpt_shape, model_shape in stats['shape_mismatch'][:20]:
                logger.warning('Shape mismatch: %s checkpoint%s vs model%s',
                               key, ckpt_shape, model_shape)
            if len(stats['shape_mismatch']) > 20:
                logger.warning('... and %d more shape-mismatched layers.',
                               len(stats['shape_mismatch']) - 20)

        if stats['missing']:
            logger.warning('Found %d model layers not loaded from %s.',
                           len(stats['missing']), load_from_meta['path'])
            for key in stats['missing'][:20]:
                logger.warning('Not loaded: %s', key)
            if len(stats['missing']) > 20:
                logger.warning('... and %d more missing layers.',
                               len(stats['missing']) - 20)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()