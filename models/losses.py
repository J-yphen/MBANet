import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GT Edge Mask Generation
# ---------------------------------------------------------------------------
def build_gt_edge_mask(
    seg_labels: torch.Tensor,
    dilation_r: int = 3,
) -> torch.Tensor:
    """
    Derives a binary edge mask from segmentation labels using morphological
    operations approximated by max-pooling.

    A pixel is "boundary" if its label differs from any neighbor within
    a radius of `dilation_r` pixels.

    Args:
        seg_labels : (B, H, W) long tensor — segmentation ground truth
        dilation_r : radius for dilation (larger = thicker boundary band)

    Returns:
        edge_mask : (B, 1, H, W) float tensor in {0.0, 1.0}
    """
    B, H, W = seg_labels.shape

    # Cast labels to float for pooling operations
    lab = seg_labels.float().unsqueeze(1)               # (B, 1, H, W)

    # Max and min pool to detect label transitions within the window
    k   = 2 * dilation_r + 1
    pad = dilation_r
    lab_max = F.max_pool2d(lab,  kernel_size=k, stride=1, padding=pad)
    # Min pool = -max(-x)
    lab_min = -F.max_pool2d(-lab, kernel_size=k, stride=1, padding=pad)

    # Where max ≠ min → boundary region
    edge_mask = (lab_max != lab_min).float()            # (B, 1, H, W)
    return edge_mask


# ---------------------------------------------------------------------------
# 1. Boundary Loss
# ---------------------------------------------------------------------------
class BoundaryLoss(nn.Module):
    """
    Binary cross-entropy between predicted boundary logits and a dilated
    GT edge mask derived from segmentation labels.

    Handles class imbalance (few boundary pixels) via positive weighting.

    Args:
        dilation_r    : dilation radius for GT edge mask thickness
        pos_weight    : upweight for positive (boundary) pixels to counter
                        class imbalance (background >> boundary)
    """

    def __init__(self, dilation_r: int = 3, pos_weight: float = 5.0):
        super().__init__()
        self.dilation_r = dilation_r
        self.pos_weight = pos_weight

    def forward(
        self,
        boundary_logits: torch.Tensor,
        seg_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            boundary_logits : (B, 1, H, W) or (B, S, H, W)
                              Raw logits (before sigmoid) from boundary head
            seg_labels      : (B, H, W) long
        Returns:
            loss : scalar tensor
        """
        B, H, W = seg_labels.shape

        # Build GT edge mask
        edge_gt = build_gt_edge_mask(seg_labels, self.dilation_r)  # (B,1,H,W)

        # If multi-scale logits (B, S, H, W), average over scale dim
        if boundary_logits.shape[1] > 1:
            boundary_logits = boundary_logits.mean(dim=1, keepdim=True)

        # Ensure spatial match
        if boundary_logits.shape[2:] != (H, W):
            boundary_logits = F.interpolate(
                boundary_logits, size=(H, W),
                mode='bilinear', align_corners=False,
            )

        # Weighted BCE
        pos_w = torch.tensor(
            [self.pos_weight], device=boundary_logits.device
        )
        loss = F.binary_cross_entropy_with_logits(
            boundary_logits,
            edge_gt,
            pos_weight=pos_w,
            reduction='mean',
        )
        return loss


# ---------------------------------------------------------------------------
# 2. Scale Entropy Regularisation
# ---------------------------------------------------------------------------
class ScaleRegLoss(nn.Module):
    """
    Penalises deviation from a target entropy level for scale attention.
    This avoids both one-hot collapse and uniform attention.
    """

    def __init__(self, target_ratio: float = 0.5):
        super().__init__()
        self.target_ratio = float(target_ratio)

    def forward(self, attn_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attn_logits : (B, S, H, W)
        Returns:
            loss : scalar — squared deviation from target entropy
        """
        S = attn_logits.shape[1]
        max_ent = math.log(float(S))
        target = self.target_ratio * max_ent

        attn = torch.softmax(attn_logits, dim=1)             # (B, S, H, W)
        entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=1)  # (B, H, W)
        loss = (entropy.mean() - target).pow(2)
        return loss


# ---------------------------------------------------------------------------
# 3. Dice Loss (foreground only)
# ---------------------------------------------------------------------------
class DiceLoss(nn.Module):
    """
    Soft Dice loss averaged over foreground classes (skips background).
    """

    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = float(smooth)

    def forward(self, seg_logits: torch.Tensor, seg_labels: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(seg_logits, dim=1)  # (B, C, H, W)
        valid = (seg_labels != self.ignore_index)
        total = 0.0

        for c in range(1, self.num_classes):
            gt_c = ((seg_labels == c) & valid).float().reshape(-1)
            pr_c = probs[:, c].reshape(-1)
            mask = valid.reshape(-1)

            inter = (pr_c[mask] * gt_c[mask]).sum()
            denom = pr_c[mask].sum() + gt_c[mask].sum()
            total += 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)

        return total / float(self.num_classes - 1)


# ---------------------------------------------------------------------------
# 4. Lovasz-Softmax Loss
# ---------------------------------------------------------------------------
def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    Computes the gradient of the Lovasz extension of the IoU loss
    for a sorted error vector.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    iou = 1.0 - intersection / union
    if p > 1:
        iou[1:p] = iou[1:p] - iou[0:-1]
    return iou


class LovaszSoftmax(nn.Module):
    """
    Lovasz-Softmax loss for multi-class segmentation.
    """

    def __init__(self, classes: str = 'present', ignore_index: int = 255):
        super().__init__()
        self.classes = classes
        self.ignore_index = ignore_index

    def _lovasz_softmax_flat(
        self,
        probs: torch.Tensor,   # (P, C)
        labels: torch.Tensor,  # (P,)
    ) -> torch.Tensor:
        C = probs.shape[1]
        losses = []
        present = labels.unique()

        for c in range(C):
            if self.classes == 'present' and c not in present:
                continue
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue
            errs = (fg - probs[:, c]).abs()
            errs_sorted, perm = torch.sort(errs, dim=0, descending=True)
            fg_sorted = fg[perm]
            losses.append(torch.dot(errs_sorted, _lovasz_grad(fg_sorted)))

        if not losses:
            return probs.sum() * 0.0
        return torch.stack(losses).mean()

    def forward(
        self,
        seg_logits: torch.Tensor,  # (B, C, H, W)
        seg_labels: torch.Tensor,  # (B, H, W)
    ) -> torch.Tensor:
        probs = torch.softmax(seg_logits, dim=1)
        B, C, H, W = probs.shape

        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)
        labels_flat = seg_labels.reshape(-1)

        valid = labels_flat != self.ignore_index
        probs_flat = probs_flat[valid]
        labels_flat = labels_flat[valid]

        return self._lovasz_softmax_flat(probs_flat, labels_flat)


# ---------------------------------------------------------------------------
# 5. Combined Loss (drop-in replacement for MBANet's criterion)
# ---------------------------------------------------------------------------
class CombinedLoss(nn.Module):
    """
    Combines:
        L_total = L_CE + λ_dice * L_dice + λ_lovasz * L_lovasz
                  + λ_bound * L_boundary + λ_scale * L_scale_reg

    Args:
        num_classes    : number of segmentation classes
        lambda_bound   : weight for boundary loss (0.4 is a safe start)
        lambda_scale   : weight for scale entropy reg (0.05)
        lambda_dice    : weight for Dice loss (0.5)
        lambda_lovasz  : weight for Lovasz-Softmax loss (0.75)
        dilation_r     : GT boundary dilation radius
        pos_weight     : BCE positive class weight
        ignore_index   : label index to ignore in CE (default 255)
    """

    def __init__(
        self,
        num_classes: int,
        lambda_bound: float = 0.4,
        lambda_scale: float = 0.05,
        lambda_dice: float = 0.5,
        lambda_lovasz: float = 0.75,
        dilation_r: int = 3,
        pos_weight: float = 3.0,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.lambda_bound = lambda_bound
        self.lambda_scale = lambda_scale
        self.lambda_dice = lambda_dice
        self.lambda_lovasz = lambda_lovasz

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.lovasz = LovaszSoftmax(classes='present', ignore_index=ignore_index)
        self.boundary_loss = BoundaryLoss(
            dilation_r=dilation_r, pos_weight=pos_weight
        )
        self.scale_reg = ScaleRegLoss()

    def forward(
        self,
        seg_logits: torch.Tensor,
        seg_labels: torch.Tensor,
        boundary_logits: torch.Tensor = None,
        attn_logits: torch.Tensor = None,
        lambda_bound: float = None,
        lambda_scale: float = None,
    ):
        """
        Args:
            seg_logits      : (B, num_classes, H, W)
            seg_labels      : (B, H, W) long
            boundary_logits : (B, 1, H, W) or (B, S, H, W) — optional
            attn_logits     : (B, S, H, W) — optional, for scale reg
            lambda_bound    : override for boundary loss weight
            lambda_scale    : override for scale regularization weight

        Returns:
            total_loss : scalar
            loss_dict  : dict with individual loss values for logging
        """
        loss_dict = {}
        eff_lambda_bound = self.lambda_bound if lambda_bound is None else float(lambda_bound)
        eff_lambda_scale = self.lambda_scale if lambda_scale is None else float(lambda_scale)

        # ── Segmentation CE ───────────────────────────────────────────────
        l_ce = self.ce_loss(seg_logits, seg_labels)
        loss_dict['ce'] = l_ce.item()
        total = l_ce

        # ── Dice loss (foreground only) ──────────────────────────────────
        l_dice = self.dice_loss(seg_logits, seg_labels)
        loss_dict['dice'] = l_dice.item()
        total = total + self.lambda_dice * l_dice

        # ── Lovasz-Softmax loss ─────────────────────────────────────────
        l_lovasz = self.lovasz(seg_logits, seg_labels)
        loss_dict['lovasz'] = l_lovasz.item()
        total = total + self.lambda_lovasz * l_lovasz

        # ── Boundary loss ─────────────────────────────────────────────────
        if boundary_logits is not None:
            l_bound = self.boundary_loss(boundary_logits, seg_labels)
            loss_dict['boundary'] = l_bound.item()
            total = total + eff_lambda_bound * l_bound

        # ── Scale entropy regularisation ──────────────────────────────────
        if attn_logits is not None:
            l_scale = self.scale_reg(attn_logits)
            loss_dict['scale_reg'] = l_scale.item()
            total = total + eff_lambda_scale * l_scale

        loss_dict['lambda_bound'] = eff_lambda_bound
        loss_dict['lambda_scale'] = eff_lambda_scale
        loss_dict['total'] = total.item()
        return total, loss_dict