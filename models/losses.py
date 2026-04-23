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
    Entropy regulariser on scale attention logits.
    Prevents all attention collapsing to one scale (degenerate solution).

    Maximises entropy of the attention distribution → encourages the network
    to genuinely use multiple scales rather than ignoring S-1 of them.

    Loss = -mean(entropy(softmax(attn_logits, dim=1)))

    Args:
        None
    """

    def __init__(self):
        super().__init__()

    def forward(self, attn_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attn_logits : (B, S, H, W)
        Returns:
            loss : scalar — negative entropy (minimising this maximises entropy)
        """
        # Softmax over scale dim
        attn = torch.softmax(attn_logits, dim=1)        # (B, S, H, W)
        # Entropy: -sum(p * log(p + eps)) over scale dim
        entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=1)  # (B, H, W)
        # We MAXIMISE entropy → loss = NEGATIVE entropy
        loss = -entropy.mean()
        return loss


# ---------------------------------------------------------------------------
# 3. Combined Loss (drop-in replacement for COSNet's criterion)
# ---------------------------------------------------------------------------
class CombinedLoss(nn.Module):
    """
    Combines:
        L_total = L_CE + λ_bound * L_boundary + λ_scale * L_scale_reg

    Args:
        num_classes    : number of segmentation classes
        lambda_bound   : weight for boundary loss (0.4 is a safe start)
        lambda_scale   : weight for scale entropy reg (0.05)
        dilation_r     : GT boundary dilation radius
        pos_weight     : BCE positive class weight
        ignore_index   : label index to ignore in CE (default 255)
    """

    def __init__(
        self,
        num_classes: int,
        lambda_bound: float = 0.4,
        lambda_scale: float = 0.05,
        dilation_r: int = 3,
        pos_weight: float = 5.0,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.lambda_bound = lambda_bound
        self.lambda_scale = lambda_scale

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
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
    ):
        """
        Args:
            seg_logits      : (B, num_classes, H, W)
            seg_labels      : (B, H, W) long
            boundary_logits : (B, 1, H, W) or (B, S, H, W) — optional
            attn_logits     : (B, S, H, W) — optional, for scale reg

        Returns:
            total_loss : scalar
            loss_dict  : dict with individual loss values for logging
        """
        loss_dict = {}

        # ── Segmentation CE ───────────────────────────────────────────────
        l_ce = self.ce_loss(seg_logits, seg_labels)
        loss_dict['ce'] = l_ce.item()
        total = l_ce

        # ── Boundary loss ─────────────────────────────────────────────────
        if boundary_logits is not None:
            l_bound = self.boundary_loss(boundary_logits, seg_labels)
            loss_dict['boundary'] = l_bound.item()
            total = total + self.lambda_bound * l_bound

        # ── Scale entropy regularisation ──────────────────────────────────
        if attn_logits is not None:
            l_scale = self.scale_reg(attn_logits)
            loss_dict['scale_reg'] = l_scale.item()
            total = total + self.lambda_scale * l_scale

        loss_dict['total'] = total.item()
        return total, loss_dict