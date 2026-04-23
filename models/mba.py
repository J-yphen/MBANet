import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LayerNorm that works on channels-first tensors (B, C, H, W)
# Identical to the one used in ConvNeXt — safe to reuse if already in repo
# ---------------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    """
    LayerNorm for (B, C, H, W) tensors.
    Standard nn.LayerNorm expects (B, H, W, C), so we permute.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)                          # channel mean
        s = (x - u).pow(2).mean(1, keepdim=True)             # channel variance
        x = (x - u) / torch.sqrt(s + self.eps)               # normalize
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# ---------------------------------------------------------------------------
# Core MBA Module
# ---------------------------------------------------------------------------
class MBA(nn.Module):
    """
    Multi-Scale Boundary Attention Module.

    Args:
        dim         : number of input/output channels (must match BEM's channel)
        pool_scales : tuple of average-pooling kernel sizes defining the
                      low-pass filters. Three scales cover thin outlines (2),
                      medium blobs (4), and coarse regions (8).
        reduction   : channel reduction ratio inside scale_attn (default 4)
    """

    def __init__(
        self,
        dim: int,
        pool_scales: tuple = (2, 4, 8),
        reduction: int = 4,
    ):
        super().__init__()
        self.pool_scales = pool_scales
        S = len(pool_scales)

        # ------------------------------------------------------------------
        # One depthwise conv per scale to refine the raw boundary residual.
        # Depthwise keeps it cheap; each conv only processes its own scale.
        # ------------------------------------------------------------------
        self.edge_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1,
                          groups=dim, bias=False),          # depthwise
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),  # pointwise
                nn.BatchNorm2d(dim),
            )
            for _ in pool_scales
        ])

        # ------------------------------------------------------------------
        # Scale attention: given F, predict which scale matters where.
        # Output is (B, S, H, W) — one weight map per scale per pixel.
        # Uses a bottleneck to keep cost low.
        # ------------------------------------------------------------------
        mid = max(dim // reduction, S)                       # bottleneck width
        self.scale_attn = nn.Sequential(
            nn.Conv2d(dim, mid, kernel_size=1, bias=False),  # compress
            nn.GELU(),
            nn.Conv2d(mid, S,  kernel_size=1, bias=True),    # S logit maps
        )

        # ------------------------------------------------------------------
        # Dilated context conv: sees a 5×5 effective receptive field cheaply
        # to give the edge signal some spatial context before gating.
        # ------------------------------------------------------------------
        self.ctx_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=2,
                      dilation=2, groups=dim, bias=False),   # depthwise dilated
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),  # pointwise
            nn.BatchNorm2d(dim),
        )

        # Gating: modulate context by edge confidence
        self.edge_gate = nn.Conv2d(dim, dim, kernel_size=1)

        # ------------------------------------------------------------------
        # Fusion: cat(x, gated_ctx, weighted_edge) → dim
        # ------------------------------------------------------------------
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.norm = LayerNorm2d(dim)
        self.act  = nn.GELU()

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation — zero-init the fusion so MBA starts as
    # identity (safe plug-in without destabilising pretrained backbone)
    # ------------------------------------------------------------------
    def _init_weights(self):
        nn.init.constant_(self.fuse[0].weight, 0)
        nn.init.constant_(self.edge_gate.bias,  0)
        nn.init.constant_(self.edge_gate.weight, 0)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Boundary residual: high-pass filter via pooling subtraction
    # ------------------------------------------------------------------
    def _boundary_residual(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """
        x     : (B, C, H, W)
        scale : pooling kernel / stride size
        returns high-frequency residual (B, C, H, W)
        """
        k   = max(2, int(scale))
        low = F.avg_pool2d(x, kernel_size=k, stride=k)              # downsample
        low = F.interpolate(low, size=x.shape[2:],
                            mode='bilinear', align_corners=False)    # upsample
        return x - low                                               # edges remain

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, C, H, W)  — feature map from backbone stage

        Returns:
            out            : (B, C, H, W)  — enhanced feature map
            attn_logits    : (B, S, H, W)  — raw scale attention (for aux loss)
            residuals      : list of S tensors (B, C, H, W) — per-scale edges
        """
        B, C, H, W = x.shape

        # ── Step 1: compute per-scale boundary residuals ──────────────────
        residuals = []
        for scale, conv in zip(self.pool_scales, self.edge_convs):
            raw_edge = self._boundary_residual(x, scale)   # (B,C,H,W)
            refined   = self.act(conv(raw_edge))            # (B,C,H,W)
            residuals.append(refined)

        # ── Step 2: predict per-scale attention weights ───────────────────
        attn_logits = self.scale_attn(x)                    # (B, S, H, W)
        attn        = torch.softmax(attn_logits, dim=1)     # sum-to-1 over scales

        # ── Step 3: attention-weighted aggregation ────────────────────────
        # stack: (B, S, C, H, W)
        stacked      = torch.stack(residuals, dim=1)
        # attn unsqueeze: (B, S, 1, H, W) → broadcast over C
        weighted_edge = (stacked * attn.unsqueeze(2)).sum(dim=1)  # (B,C,H,W)

        # ── Step 4: dilated context + edge-conditioned gate ───────────────
        ctx  = self.act(self.ctx_conv(x))                   # (B,C,H,W)
        gate = torch.sigmoid(self.edge_gate(weighted_edge)) # (B,C,H,W) ∈(0,1)

        # ── Step 5: fuse three streams ────────────────────────────────────
        fused = torch.cat([x, ctx * gate, weighted_edge], dim=1)  # (B,3C,H,W)
        out   = self.act(self.fuse(fused))                  # (B,C,H,W)
        out   = self.norm(out)

        return out, attn_logits, residuals