import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from mbanet_modules import FSB, LayerNorm
from mmseg.models.builder import BACKBONES

# ── NEW: import MBA ──────────────────────────────────────────────────────────
from models.mba import MBA


@BACKBONES.register_module()
class MBANet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, img_size=224,
                 depths=[3, 3, 12, 3], dim=72, expan_ratio=4, num_stages=4,
                 s_kernel_size=[5, 5, 3, 3], drop_path_rate=0.2,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 # ── NEW: MBA hyper-params (safe defaults, no breaking change)
                 mba_pool_scales=(2, 4, 8),
                 mba_reduction=4,
                 **kwargs):
        super().__init__()

        self.num_stages  = num_stages
        self.num_classes = num_classes

        # ── Channel dims per stage: [72, 144, 288, 576] for dim=72 ────────
        self.dims = [dim * (2 ** ii) for ii in range(self.num_stages)]

        # ── Stem + 3 intermediate downsampling convs ───────────────────────
        self.downsample_layers = nn.ModuleList()
        stem = nn.Conv2d(in_chans, self.dims[0],
                         kernel_size=5, stride=4, padding=2)
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(
                nn.Conv2d(self.dims[i], self.dims[i + 1],
                          kernel_size=3, stride=2, padding=1)
            )

        # ── 4 Feature stages (FSB blocks) ─────────────────────────────────
        self.stages   = nn.ModuleList()
        dp_rates      = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur           = 0
        for i in range(self.num_stages):
            stage_blocks = [
                FSB(
                    dim=self.dims[i],
                    s_kernel_size=s_kernel_size[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                    expan_ratio=expan_ratio,
                )
                for j in range(depths[i])
            ]
            self.stages.append(nn.Sequential(*stage_blocks))
            cur += depths[i]

        # ── OLD: self.hdr_layer = BEM(self.dims[-2])
        # ── NEW: MBA replaces BEM on Stage 3 features (self.dims[-2])
        # dims[-2] = dim * 4 = 288 for default dim=72
        self.hdr_layer = MBA(
            dim=self.dims[-2],          # same channel arg BEM received
            pool_scales=mba_pool_scales,
            reduction=mba_reduction,
        )

        self.apply(self._init_weights)

    # ── Weight init (unchanged) ────────────────────────────────────────────
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ── Pretrained weight loading (unchanged) ─────────────────────────────
    def init_weights(self, pretrained=None):
        if pretrained is not None:
            cur_state_dict = self.state_dict()
            checkpoint     = torch.load(pretrained, map_location="cpu")
            loaded_dict    = checkpoint.get("state_dict", checkpoint)

            new_state_dict = {}
            for key, val in loaded_dict.items():
                new_key = f"backbone.{key}" if f"backbone.{key}" in cur_state_dict else key

                if new_key not in cur_state_dict:
                    # ── NEW: skip MBA keys not in pretrained (expected) ───
                    print(f"[init_weights] Skipping unknown key: {new_key}")
                    new_state_dict[new_key] = val
                    continue

                if val.shape != cur_state_dict[new_key].shape:
                    if val.dim() == 4:
                        val = F.interpolate(
                            val,
                            size=cur_state_dict[new_key].shape[2:],
                            mode='bilinear',
                            align_corners=True,
                        )
                    else:
                        print(f"[init_weights] Shape mismatch, skipping: "
                              f"{new_key} {val.shape} vs {cur_state_dict[new_key].shape}")
                        continue

                new_state_dict[new_key] = val

            msg = self.load_state_dict(new_state_dict, strict=False)
            print(msg)

    # ── Feature extraction (unchanged) ────────────────────────────────────
    def forward_features(self, x):
        feats = []
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feats.append(x)
        return feats

    # ── Forward (MBA wired in, boundary outputs stored for loss) ──────────
    def forward(self, x):
        f1, f2, f3, f4 = self.forward_features(x)

        # ── OLD: return [f1, f2, f3, f4]
        # ── NEW: apply MBA to f3 (Stage 3), same as original BEM position
        f3_enhanced, attn_logits, residuals, boundary_logits = self.hdr_layer(f3)

        # Store auxiliary outputs as instance attributes so the
        # decode head / loss can read them without changing the
        # return signature (decoder still sees 4 feature maps).
        self.boundary_logits      = boundary_logits  # (B, 1, H, W)
        self.attn_logits          = attn_logits      # (B, S, H, W)
        self.boundary_attn_logits = attn_logits      # (B, S, H, W) compat
        self.boundary_residuals   = residuals        # list of S × (B,C,H,W)

        return [f1, f2, f3_enhanced, f4]


@BACKBONES.register_module()
class SegNet(MBANet):
    """Compatibility alias for configs that reference backbone type `SegNet`."""
    pass


###############################################################################
if __name__ == "__main__":
    model = MBANet(
        in_chans=3, num_classes=1000, img_size=224,
        depths=[3, 3, 12, 3], dim=72, expan_ratio=4, num_stages=4,
        s_kernel_size=[5, 5, 3, 3], drop_path_rate=0.2,
        layer_scale_init_value=1e-6, head_init_scale=1.,
        # MBA specific — optional, defaults are fine
        mba_pool_scales=(2, 4, 8),
        mba_reduction=4,
    )

    # ── Parameter count ───────────────────────────────────────────────────
    def count_parameters(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    total = count_parameters(model)
    print(f"Total Trainable Params: {round(total * 1e-6, 2)} M")

    # ── Shape sanity check ────────────────────────────────────────────────
    dummy = torch.randn(2, 3, 224, 224)
    outs  = model(dummy)
    names = ['f1', 'f2', 'f3_enhanced', 'f4']
    for name, feat in zip(names, outs):
        print(f"  {name}: {feat.shape}")

    # ── Auxiliary outputs ─────────────────────────────────────────────────
    print(f"  attn_logits : {model.attn_logits.shape}")
    print(f"  boundary    : {model.boundary_logits.shape}")
    print(f"  residuals   : {len(model.boundary_residuals)} × {model.boundary_residuals[0].shape}")

    # ── FLOPs ─────────────────────────────────────────────────────────────
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    inp   = torch.ones(1, 3, 224, 224)
    flops = FlopCountAnalysis(model, inp)
    print(flop_count_table(flops))