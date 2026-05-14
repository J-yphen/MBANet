import argparse
import os
import sys
from pathlib import Path

import cv2
import mmcv
import numpy as np
from mmcv.runner import load_checkpoint
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.datasets import build_dataset


def _load_module_root(config_path):
    config_path = str(config_path)
    if "/prog_lab_mbanet/" in config_path:
        return "prog_lab_mbanet"
    if "/mbanet/" in config_path:
        return None
    return None


def _register_custom_modules(project_root, module_root):
    module_path = project_root if module_root is None else project_root / module_root
    if not module_path.exists():
        module_path = project_root
    sys.path.insert(0, str(module_path))

    # Dataset + pipeline components.
    import align_resize  # noqa: F401
    import zerowaste  # noqa: F401
    import specwaste  # noqa: F401

    # Custom segmentor + MBANet backbone.
    import models  # noqa: F401
    import mbanet  # noqa: F401



def _apply_palette(mask, palette):
    palette = np.asarray(palette, dtype=np.uint8)
    color = palette[mask]
    return color


def _save_results(mask, palette, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"{tag}_mask.png")
    color_path = os.path.join(out_dir, f"{tag}_color.png")

    mmcv.imwrite(mask.astype(np.uint8), raw_path)
    if palette is not None:
        color = _apply_palette(mask, palette)
        mmcv.imwrite(color, color_path)

    return raw_path, color_path


def _make_overlay(image_bgr, color_mask_bgr, alpha=0.5):
    image = image_bgr.astype(np.float32)
    mask = color_mask_bgr.astype(np.float32)
    blended = image * (1.0 - alpha) + mask * alpha
    return blended.astype(np.uint8)


def _infer_gt_path(image_path):
    image_path = str(image_path)
    if "/data/" in image_path:
        return image_path.replace("/data/", "/sem_seg/")
    return None


def _label_tile(tile, label, label_h=48, pad=10):
    h, w = tile.shape[:2]
    canvas = np.full((h + label_h, w, 3), 255, dtype=np.uint8)
    canvas[label_h:, :w] = tile

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_x = max(pad, (w - text_size[0]) // 2)
    text_y = (label_h + text_size[1]) // 2
    cv2.putText(canvas, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return canvas


def _make_panel(tiles, labels):
    heights = [tile.shape[0] for tile in tiles]
    max_h = max(heights)
    padded = []
    for tile, label in zip(tiles, labels):
        if tile.shape[0] < max_h:
            pad = np.full((max_h - tile.shape[0], tile.shape[1], 3), 255, dtype=np.uint8)
            tile = np.vstack([tile, pad])
        padded.append(_label_tile(tile, label))
    return np.concatenate(padded, axis=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Single-image segmentation inference")
    parser.add_argument("--config", required=True, help="Path to model config")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--tag",
        default=None,
        help="Output name prefix (default: model folder name)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for inference, e.g., cuda:0 or cpu",
    )
    parser.add_argument(
        "--gt",
        default=None,
        help="Optional ground-truth mask path (defaults to /data/ -> /sem_seg/)",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.5,
        help="Alpha for overlay blending in (0, 1]",
    )
    parser.add_argument(
        "--only-overlays",
        action="store_true",
        help="Render only input, predicted overlay, and GT overlay",
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Save predicted and GT overlays as individual images",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    module_root = _load_module_root(args.config)
    _register_custom_modules(project_root, module_root)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    # Ensure PALETTE/CLASSES are available even if checkpoint meta is missing.
    if "PALETTE" not in cfg and "CLASSES" not in cfg:
        dataset = build_dataset(cfg.data.test)
        cfg.PALETTE = dataset.PALETTE
        cfg.CLASSES = dataset.CLASSES

    model = init_segmentor(cfg, checkpoint=None, device=args.device)
    load_checkpoint(model, args.checkpoint, map_location="cpu", strict=False)
    if not hasattr(model, "PALETTE") and "PALETTE" in cfg:
        model.PALETTE = cfg.PALETTE
    if not hasattr(model, "CLASSES") and "CLASSES" in cfg:
        model.CLASSES = cfg.CLASSES
    result = inference_segmentor(model, args.image)
    mask = result[0]

    tag = args.tag
    if tag is None:
        tag = Path(args.config).stem

    palette = getattr(model, "PALETTE", None)
    raw_path, color_path = _save_results(mask, palette, args.out_dir, tag)

    if palette is None:
        print("Palette not found; skipping composite panel.")
        return

    image_bgr = mmcv.imread(args.image)
    pred_color = _apply_palette(mask, palette)
    pred_overlay = _make_overlay(image_bgr, pred_color, alpha=args.overlay_alpha)

    gt_path = args.gt or _infer_gt_path(args.image)
    if gt_path is None or not os.path.exists(gt_path):
        print("Ground truth not found; skipping composite panel.")
        return

    gt_mask = mmcv.imread(gt_path, flag="unchanged")
    gt_mask = gt_mask.astype(np.int64)
    gt_color = _apply_palette(gt_mask, palette)
    gt_overlay = _make_overlay(image_bgr, gt_color, alpha=args.overlay_alpha)

    if args.only_overlays:
        tiles = [image_bgr, pred_overlay, gt_overlay]
        labels = [
            "Real image",
            "Predicted overlay",
            "GT overlay",
        ]
    else:
        tiles = [image_bgr, pred_color, pred_overlay, gt_color, gt_overlay]
        labels = [
            "Real image",
            "Predicted mask",
            "Predicted overlay",
            "Ground truth",
            "GT overlay",
        ]
    if args.save_overlays:
        pred_overlay_path = os.path.join(args.out_dir, f"{tag}_pred_overlay.png")
        gt_overlay_path = os.path.join(args.out_dir, f"{tag}_gt_overlay.png")
        mmcv.imwrite(pred_overlay, pred_overlay_path)
        mmcv.imwrite(gt_overlay, gt_overlay_path)
        print(f"Saved predicted overlay: {pred_overlay_path}")
        print(f"Saved GT overlay: {gt_overlay_path}")

    panel = _make_panel(tiles, labels)
    panel_path = os.path.join(args.out_dir, f"{tag}_panel.png")
    mmcv.imwrite(panel, panel_path)
    print(f"Saved panel: {panel_path}")

    print(f"Saved mask: {raw_path}")
    if palette is not None:
        print(f"Saved color mask: {color_path}")


if __name__ == "__main__":
    main()
