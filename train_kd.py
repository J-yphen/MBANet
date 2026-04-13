import argparse
import os
import os.path as osp
import random
import sys
import time
from datetime import timedelta

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

# Ensure local modules are importable when script is launched from other cwd.
SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Register custom backbones, transforms, and datasets.
import cosnet  # noqa: F401,E402
import student_cosnet  # noqa: F401,E402
from align_resize import AlignResize  # noqa: F401,E402
from specwaste import SpectralWasteDataset  # noqa: F401,E402
from zerowaste import ZeroWasteDataset  # noqa: F401,E402

def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge distillation training for segmentation")
    parser.add_argument("--teacher-config", required=True, help="Teacher mmseg config")
    parser.add_argument(
        "--teacher-ckpt",
        default=None,
        help="Teacher checkpoint path. If omitted, uses model.pretrained from teacher config",
    )
    parser.add_argument("--student-config", required=True, help="Student mmseg config")
    parser.add_argument("--student-init", default=None, help="Optional student initialization checkpoint")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume checkpoint path for exact KD state restoration (student/optimizer/iter/rng)",
    )
    parser.add_argument("--work-dir", default="kd_logs", help="Directory for distilled checkpoints")
    parser.add_argument("--max-iters", type=int, default=40000, help="Total training iterations")
    parser.add_argument("--save-interval", type=int, default=4000, help="Checkpoint interval")
    parser.add_argument("--val-interval", type=int, default=1000, help="Validation interval for mIoU")
    parser.add_argument("--lr", type=float, default=9e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="AdamW weight decay")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--hard-weight", type=float, default=1.0, help="Hard CE loss weight")
    parser.add_argument("--soft-weight", type=float, default=1.0, help="Soft KL loss weight")
    parser.add_argument("--feat-weight", type=float, default=0.5, help="Feature loss weight")
    parser.add_argument("--boundary-weight", type=float, default=0.2, help="Boundary loss weight for BEM-based student")
    parser.add_argument(
        "--boundary-loss-iters",
        type=int,
        default=40000,
        help="Apply dedicated boundary loss until this iteration (inclusive)",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Training device")
    return parser.parse_args()


def _unwrap_data(value):
    if hasattr(value, "data"):
        value = value.data[0]
    while isinstance(value, (list, tuple)) and len(value) > 0:
        value = value[0]
    return value


def _kd_kl_div(student_logits, teacher_logits, temperature):
    # For segmentation logits [B, C, H, W], batchmean scales by 1/B only and
    # can explode with image size. Use elementwise mean for stable magnitude.
    log_p = F.log_softmax(student_logits / temperature, dim=1)
    q = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(log_p, q, reduction="mean") * (temperature ** 2)


def _build_boundary_target(gt):
    # gt: [B, H, W] with ignore label 255
    valid = (gt != 255)
    edge = torch.zeros_like(gt, dtype=torch.float32)

    h_diff = (gt[:, 1:, :] != gt[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    w_diff = (gt[:, :, 1:] != gt[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]

    edge[:, 1:, :] = torch.maximum(edge[:, 1:, :], h_diff.float())
    edge[:, :-1, :] = torch.maximum(edge[:, :-1, :], h_diff.float())
    edge[:, :, 1:] = torch.maximum(edge[:, :, 1:], w_diff.float())
    edge[:, :, :-1] = torch.maximum(edge[:, :, :-1], w_diff.float())

    # Slightly thicken boundaries for stable supervision.
    edge = F.max_pool2d(edge.unsqueeze(1), kernel_size=3, stride=1, padding=1)
    return edge, valid.float().unsqueeze(1)


def _boundary_loss(boundary_logits, gt):
    if not boundary_logits:
        return gt.new_tensor(0.0, dtype=torch.float32)

    boundary_target, valid_mask = _build_boundary_target(gt)
    total = gt.new_tensor(0.0, dtype=torch.float32)
    used = 0
    for logit in boundary_logits:
        tgt = F.interpolate(boundary_target, size=logit.shape[-2:], mode="nearest")
        msk = F.interpolate(valid_mask, size=logit.shape[-2:], mode="nearest")
        raw = F.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
        denom = msk.sum().clamp_min(1.0)
        total = total + (raw * msk).sum() / denom
        used += 1
    return total / max(used, 1)


def _resolve_teacher_checkpoint(args, teacher_cfg):
    if args.teacher_ckpt is not None:
        return args.teacher_ckpt

    ckpt_path = teacher_cfg.model.get("pretrained", None)
    if ckpt_path is None:
        raise ValueError("No teacher checkpoint provided. Use --teacher-ckpt or set model.pretrained in teacher config.")
    return ckpt_path


def _resolve_config_path(path_arg):
    if osp.isfile(path_arg):
        return path_arg

    script_dir = osp.dirname(osp.abspath(__file__))
    candidates = [
        osp.join(script_dir, path_arg),
        osp.join(script_dir, "configs", "cosnet", path_arg),
        osp.join(script_dir, "configs", path_arg),
    ]

    for cand in candidates:
        if osp.isfile(cand):
            return cand

    raise FileNotFoundError(
        f'Could not find config file: "{path_arg}". '
        f'Tried: {path_arg}, ' + ", ".join(candidates)
    )


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not isinstance(state, dict):
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _load_resume_checkpoint(resume_path, student, feat_align, optimizer, device):
    ckpt = torch.load(resume_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Resume checkpoint must be a dict. Got: {type(ckpt)}")

    if "student" in ckpt:
        msg = student.load_state_dict(ckpt["student"], strict=False)
        print(f"Resumed student weights: {msg}")
    else:
        raise KeyError("Resume checkpoint missing key: 'student'")

    if "feat_align" in ckpt:
        feat_align.load_state_dict(ckpt["feat_align"], strict=False)
    else:
        raise KeyError("Resume checkpoint missing key: 'feat_align'")

    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        # Move optimizer tensors to current device.
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    else:
        raise KeyError("Resume checkpoint missing key: 'optimizer'")

    start_iter = int(ckpt.get("iteration", 0)) + 1
    best_miou = float(ckpt.get("best_miou", -1.0))
    if "rng_state" in ckpt:
        _restore_rng_state(ckpt["rng_state"])

    return start_iter, best_miou


@torch.no_grad()
def evaluate_miou(student, data_loader, device, num_classes):
    student.eval()
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    gt_iter = iter(data_loader.dataset.get_gt_seg_maps())

    for batch in data_loader:
        imgs = _unwrap_data(batch["img"]).to(device, non_blocking=True)

        # Use explicit backbone+decode forward to avoid mmseg version-specific
        # return-type differences from encode_decode/simple_test.
        feats = student.backbone(imgs)
        logits = student.decode_head.forward(feats)

        bsz = logits.size(0)
        for b in range(bsz):
            gt_np = next(gt_iter)
            gt = torch.from_numpy(gt_np).long().to(device, non_blocking=True)

            sample_logits = logits[b:b + 1]
            sample_logits = F.interpolate(sample_logits, size=gt.shape[-2:], mode="bilinear", align_corners=False)
            pred = sample_logits.argmax(dim=1).squeeze(0)

            valid = gt != 255
            for cls in range(num_classes):
                pred_c = (pred == cls) & valid
                gt_c = (gt == cls) & valid
                intersection[cls] += (pred_c & gt_c).sum()
                union[cls] += (pred_c | gt_c).sum()

    iou = intersection / (union + 1e-6)
    miou = iou.mean().item() * 100.0
    student.train()
    return miou


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    teacher_cfg_path = _resolve_config_path(args.teacher_config)
    student_cfg_path = _resolve_config_path(args.student_config)
    print(f"Resolved teacher config: {teacher_cfg_path}")
    print(f"Resolved student config: {student_cfg_path}")

    teacher_cfg = mmcv.Config.fromfile(teacher_cfg_path)
    student_cfg = mmcv.Config.fromfile(student_cfg_path)
    teacher_ckpt = _resolve_teacher_checkpoint(args, teacher_cfg)

    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))

    teacher = build_segmentor(teacher_cfg.model, train_cfg=None, test_cfg=teacher_cfg.get("test_cfg"))
    load_checkpoint(teacher, teacher_ckpt, map_location="cpu", strict=False)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = build_segmentor(student_cfg.model, train_cfg=student_cfg.get("train_cfg"), test_cfg=student_cfg.get("test_cfg"))
    if args.resume_from is not None and args.student_init is not None:
        raise ValueError("Use either --student-init or --resume-from, not both.")

    if args.student_init is not None:
        load_checkpoint(student, args.student_init, map_location="cpu", strict=False)
    student.to(device)
    student.train()

    train_dataset = build_dataset(student_cfg.data.train)
    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=student_cfg.data.samples_per_gpu,
        workers_per_gpu=student_cfg.data.workers_per_gpu,
        dist=False,
        shuffle=True,
    )

    val_dataset = build_dataset(student_cfg.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=student_cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    teacher_dims = teacher.backbone.dims
    student_dims = student.backbone.dims
    feat_align = torch.nn.ModuleList(
        [
            torch.nn.Conv2d(student_dims[i], teacher_dims[i], kernel_size=1, bias=False)
            for i in range(min(len(student_dims), len(teacher_dims)))
        ]
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(feat_align.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    start_iter = 1
    best_miou = -1.0
    if args.resume_from is not None:
        start_iter, best_miou = _load_resume_checkpoint(
            args.resume_from,
            student,
            feat_align,
            optimizer,
            device,
        )
        print(f"Resumed from: {args.resume_from}")
        print(f"Restarting at iteration: {start_iter}")
        print(f"Recovered best mIoU: {best_miou:.2f}")

    data_iter = iter(train_loader)
    start_time = time.time()
    num_classes = student_cfg.model.decode_head.num_classes

    for iteration in range(start_iter, args.max_iters + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        imgs = _unwrap_data(batch["img"]).to(device, non_blocking=True)
        gt = _unwrap_data(batch["gt_semantic_seg"]).long().to(device, non_blocking=True)
        if gt.dim() == 4 and gt.size(1) == 1:
            gt = gt.squeeze(1)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_feats = teacher.backbone(imgs)
            teacher_logits = teacher.decode_head.forward(teacher_feats)

        student_feats = student.backbone(imgs)
        student_logits = student.decode_head.forward(student_feats)

        teacher_logits = F.interpolate(teacher_logits, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        student_logits = F.interpolate(student_logits, size=gt.shape[-2:], mode="bilinear", align_corners=False)

        hard_loss = F.cross_entropy(student_logits, gt, ignore_index=255)
        soft_loss = _kd_kl_div(student_logits, teacher_logits.detach(), args.temperature)

        feat_loss = 0.0
        for i, proj in enumerate(feat_align):
            s_feat = proj(student_feats[i])
            t_feat = teacher_feats[i].detach()
            if s_feat.shape[-2:] != t_feat.shape[-2:]:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[-2:], mode="bilinear", align_corners=False)
            feat_loss = feat_loss + F.mse_loss(s_feat, t_feat)

        boundary_loss = gt.new_tensor(0.0, dtype=torch.float32)
        if iteration <= args.boundary_loss_iters:
            boundary_logits = getattr(student.backbone, "latest_boundary_logits", None)
            if boundary_logits is not None:
                boundary_loss = _boundary_loss(boundary_logits, gt)

        total_loss = (
            args.hard_weight * hard_loss
            + args.soft_weight * soft_loss
            + args.feat_weight * feat_loss
            + args.boundary_weight * boundary_loss
        )

        if not torch.isfinite(total_loss):
            print(
                f"Non-finite loss at iter {iteration}: "
                f"total={total_loss.item()}, hard={hard_loss.item()}, "
                f"soft={soft_loss.item()}, feat={feat_loss.item()}, "
                f"boundary={boundary_loss.item()}. Stopping training."
            )
            raise FloatingPointError("Encountered non-finite loss during KD training.")

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(student.parameters()) + list(feat_align.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        if iteration % 50 == 0 or iteration == 1:
            elapsed = time.time() - start_time
            avg_iter_time = elapsed / iteration
            eta_seconds = max(0.0, (args.max_iters - iteration) * avg_iter_time)
            eta_td = timedelta(seconds=int(eta_seconds))
            print(
                f"Iter {iteration:06d}/{args.max_iters}: "
                f"total={total_loss.item():.4f}, "
                f"hard={hard_loss.item():.4f}, "
                f"soft={soft_loss.item():.4f}, "
                f"feat={feat_loss.item():.4f}, "
                f"boundary={boundary_loss.item():.4f}, "
                f"elapsed={elapsed:.1f}s, "
                f"eta={eta_td}"
            )

        if iteration % args.save_interval == 0 or iteration == args.max_iters:
            ckpt_path = osp.join(args.work_dir, f"student_kd_iter_{iteration}.pth")
            torch.save(
                {
                    "iteration": iteration,
                    "best_miou": best_miou,
                    "student": student.state_dict(),
                    "feat_align": feat_align.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_state": _capture_rng_state(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

        if iteration % args.val_interval == 0 or iteration == args.max_iters:
            miou = evaluate_miou(student, val_loader, device, num_classes)
            print(f"Validation @ iter {iteration:06d}: mIoU={miou:.2f}")
            if miou > best_miou:
                best_miou = miou
                best_path = osp.join(args.work_dir, "student_kd_best.pth")
                torch.save(
                    {
                        "iteration": iteration,
                        "best_miou": best_miou,
                        "student": student.state_dict(),
                        "feat_align": feat_align.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rng_state": _capture_rng_state(),
                        "args": vars(args),
                    },
                    best_path,
                )
                print(f"New best checkpoint saved: {best_path}")


if __name__ == "__main__":
    main()
