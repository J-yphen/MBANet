import torch
from torch import nn
from timm.models.layers import DropPath
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MCFS(nn.Module):
    def __init__(self, dim, s_kernel_size=3):
        super().__init__()
        
        self.proj_1 = nn.Conv2d(dim, dim, 1, padding=0)
        self.proj_2 = nn.Conv2d(dim*2, dim, 1, padding=0)
        self.norm_proj = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        # multiscale spatial context layers
        self.s_ctx_1 = nn.Conv2d(dim, dim, kernel_size=s_kernel_size, padding=s_kernel_size//2, groups=dim//4)
        self.s_ctx_2 = nn.Conv2d(dim, dim, kernel_size=s_kernel_size, dilation=2, padding=(s_kernel_size//2)*2, groups=dim//4)
        self.norm_s = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        # sharpening module layers
        self.h_ctx = nn.Conv2d(dim, dim, kernel_size=5, padding=2, bias=False, groups=dim)
        self.norm_h = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        
        self.act = nn.GELU()


    def forward(self, x):
        x = self.norm_proj(self.act(self.proj_1(x)))
        B, C, H, W = x.shape

        # extract multi-scale contextual features
        sx1 = self.act(self.s_ctx_1(x))
        sx2 = self.act(self.s_ctx_2(x))

        sx = self.norm_s(sx1 + sx2)

        # feature enhancement using learnable sharpening factors
        # implementation of sharpening module
        hx = self.act(self.h_ctx(x))
        hx_t = x - hx.mean(dim=1).unsqueeze(1)
        hx_t = torch.softmax(hx.mean(dim=[-2,-1]).unsqueeze(-1).unsqueeze(-1), dim=1) * hx_t
        hx = self.norm_h(hx + hx_t)

        # combine the multiscale contetxual features with the sharpened features
        x = self.act(self.proj_2(torch.cat([sx, hx], dim=1)))

        return x


class FSB(nn.Module):
    """
    Feature Sharpening Block: 
    It is the core block of the COSNet encoder/backbone,
    utilized to extract semantically rich features for segementation task in cluttered background.
    -----------------------------------------------
    dim:           Input channel dimension
    s_kernel_size: Kernel size for spatial context layers
    expan_ratio:   Expansion ratio used for channels in MLP
    ------------------------------------------------

    """
    def __init__(self, dim, s_kernel_size=3, drop_path=0.1, layer_scale_init_value=1e-6, expan_ratio=4):
        super().__init__()

        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm_dw = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.layer_norm_1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm_2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.mlp = MLP(dim=dim, mlp_ratio=expan_ratio)
        self.attn = MCFS(dim, s_kernel_size=s_kernel_size)

        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))

        self.drop_path_1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.act = nn.GELU()
        
        
    def forward(self, x):
        B, C, H, W = x.shape

        x = x + self.norm_dw(self.act(self.conv_dw(x)))

        x_copy = x
        x = self.layer_norm_1(x_copy)
        x = self.drop_path_1(self.layer_scale_1[:, None, None] * self.attn(x))
        out = x + x_copy

        x = self.layer_norm_2(out)
        x = self.drop_path_2(self.layer_scale_2[:, None, None] * self.mlp(x))
        out = out + x

        return out


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        
        self.fc_1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc_2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.fc_1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc_2(x)

        return x


class MBA(nn.Module):
    def __init__(self, dim, pool_scales=(2, 4, 8)):
        super().__init__()
        self.pool_scales = pool_scales

        # Multi-scale low-pass subtraction branches.
        self.edge_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
            for _ in pool_scales
        ])

        # Predict per-scale attention from the feature map.
        self.scale_attn = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, len(pool_scales), kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.ctx_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, dilation=2, groups=dim, bias=False)
        self.edge_gate = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.fuse = nn.Conv2d(dim * 3, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()

    def _boundary_residual(self, x, scale):
        k = max(2, int(scale))
        low = F.avg_pool2d(x, kernel_size=k, stride=k)
        low = F.interpolate(low, size=x.shape[2:], mode='bilinear', align_corners=False)
        return x - low

    def forward(self, x):
        residuals = []
        for scale, conv in zip(self.pool_scales, self.edge_convs):
            edge = self._boundary_residual(x, scale)
            residuals.append(self.act(conv(edge)))

        # [B, S, H, W], normalized across scales.
        attn_logits = self.scale_attn(x)
        attn = torch.softmax(attn_logits, dim=1)

        stacked_edges = torch.stack(residuals, dim=1)  # [B, S, C, H, W]
        weighted_edge = (stacked_edges * attn.unsqueeze(2)).sum(dim=1)

        gate = torch.sigmoid(self.edge_gate(weighted_edge))
        ctx = self.act(self.ctx_conv(x))

        fused = torch.cat([x, ctx * gate, weighted_edge], dim=1)
        out = self.fuse(fused)
        out = self.act(out)
        out = self.norm(out)

        # Return boundary logits proxy for optional auxiliary boundary loss.
        return out, attn_logits, residuals