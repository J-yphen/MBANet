# MBANet (COSNet + MBA)

This repository contains MBANet, a semantic segmentation project that combines a COSNet-style backbone with Multi-Scale Boundary Attention (MBA) and a boundary-aware combined loss, built on top of MMsegmentation.

## Environment setup

The project expects Python 3.8 and the MMsegmentation 0.30.x stack. The commands below match the versions used in the codebase:

```bash
python 3.8

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

pip install mmengine==0.10.7 mmsegmentation==0.30.0
```

Other imports used in the code include `timm`, `numpy`, `opencv-python`, and `fvcore` (only for the optional FLOPs check in `cosnet.py` when running it as a script). Install them if needed.

## Datasets

Update `data_root` in the dataset config that you use.

### ZeroWaste

Config: [configs/_base_/datasets/zero_waste.py](configs/_base_/datasets/zero_waste.py)

Expected structure (from the config):

```
<DATA_ROOT>/
  train/
    data/
    sem_seg/
  val/
    data/
    sem_seg/
  test/
    data/
    sem_seg/
```

The dataset class is `ZeroWasteDataset` in [zerowaste.py](zerowaste.py) and uses `.PNG` as image and mask suffixes by default.

### SpectralWaste

Config: [configs/_base_/datasets/spec_waste.py](configs/_base_/datasets/spec_waste.py)

Expected structure (from the config):

```
<DATA_ROOT>/
  rgb/
    train/
    test/
  labels_rgb/
    train/
    test/
```

The dataset class is `SpectralWasteDataset` in [specwaste.py](specwaste.py) and uses `.png` suffixes by default.

## Training

Example: ZeroWaste (single GPU)

```bash
python train.py configs/cosnet/uper_cosnet_zerowaste_40k.py --work-dir zerowaste_logs
```

Example: SpectralWaste (single GPU)

```bash
python train.py configs/cosnet/uper_cosnet_specwaste_40k.py --work-dir spectralwaste_logs
```

Notes:
- The training script registers custom modules by importing `align_resize`, `zerowaste`, and `specwaste`.
- Use `--load-from` to load a checkpoint. If the checkpoint contains only backbone weights, it is used as `model.pretrained` automatically.
- `--launcher` supports `none`, `pytorch`, `slurm`, and `mpi` as in [train.py](train.py).

## Evaluation

`test.py` runs inference on the test split, computes classwise IoU and pixel accuracy, and prints a summary.

Example:

```bash
python test.py configs/cosnet/uper_cosnet_zerowaste_40k.py /path/to/checkpoint.pth --eval mIoU
```

Optional flags (see [test.py](test.py)):
- `--show-dir` to dump visualizations
- `--out` to save raw outputs as a pickle
- `--eval-options` for MMseg evaluation options

## Single-image inference

`generate_segmentation.py` runs inference on one image and writes masks and overlays.

Example:

```bash
python generate_segmentation.py \
  --config configs/cosnet/uper_cosnet_zerowaste_40k.py \
  --checkpoint /path/to/checkpoint.pth \
  --image /path/to/image.png \
  --out-dir outputs
```

Outputs (written into `--out-dir`):
- `<tag>_mask.png` raw predicted mask
- `<tag>_color.png` colorized mask (if palette is available)
- `<tag>_panel.png` side-by-side panel (input, mask, overlays)

You can pass `--gt` to provide a ground-truth mask. If omitted, the script tries to replace `/data/` with `/sem_seg/` in the image path.
s