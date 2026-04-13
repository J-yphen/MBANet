import torch
from torch import nn
from timm.models.layers import trunc_normal_

from cosnet_modules import BEM, FSB, LayerNorm
from mmseg.models.builder import BACKBONES


class _BasicConvStage(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


@BACKBONES.register_module()
class StudentCOSNet(nn.Module):
    """Lightweight COSNet backbone for knowledge distillation."""

    def __init__(
        self,
        in_chans=3,
        depths=[2, 2, 6, 2],
        dim=48,
        expan_ratio=4,
        num_stages=4,
        s_kernel_size=[5, 5, 3, 3],
        drop_path_rate=0.1,
        layer_scale_init_value=1e-6,
        **kwargs,
    ):
        super().__init__()

        self.num_stages = num_stages
        self.dims = [dim * (2 ** i) for i in range(self.num_stages)]

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Conv2d(in_chans, self.dims[0], kernel_size=5, stride=4, padding=2))
        for i in range(self.num_stages - 1):
            self.downsample_layers.append(
                nn.Conv2d(self.dims[i], self.dims[i + 1], kernel_size=3, stride=2, padding=1)
            )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stages):
            blocks = []
            for j in range(depths[i]):
                blocks.append(
                    FSB(
                        dim=self.dims[i],
                        s_kernel_size=s_kernel_size[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        expan_ratio=expan_ratio,
                    )
                )
            self.stages.append(nn.Sequential(*blocks))
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return

        cur_state_dict = self.state_dict()
        checkpoint = torch.load(pretrained, map_location="cpu")
        loaded_dict = checkpoint.get("state_dict", checkpoint)

        new_state_dict = {}
        for key, val in loaded_dict.items():
            new_key = f"backbone.{key}" if f"backbone.{key}" in cur_state_dict else key
            if new_key not in cur_state_dict:
                continue
            if val.shape != cur_state_dict[new_key].shape:
                continue
            new_state_dict[new_key] = val

        self.load_state_dict(new_state_dict, strict=False)

    def forward_features(self, x):
        feats = []
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feats.append(x)
        return feats

    def forward(self, x):
        return self.forward_features(x)


@BACKBONES.register_module()
class BasicStudentNet(nn.Module):
    """Very basic 4-stage CNN backbone compatible with UPerNet."""

    def __init__(
        self,
        in_chans=3,
        dims=[32, 64, 128, 256],
        **kwargs,
    ):
        super().__init__()
        self.dims = dims
        self.num_stages = 4

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True),
        )

        # Output strides are 4, 8, 16, 32 for UPerNet compatibility.
        self.stage1 = _BasicConvStage(dims[0], dims[0], stride=2)
        self.stage2 = _BasicConvStage(dims[0], dims[1], stride=2)
        self.stage3 = _BasicConvStage(dims[1], dims[2], stride=2)
        self.stage4 = _BasicConvStage(dims[2], dims[3], stride=2)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm, nn.BatchNorm2d)):
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return

        cur_state_dict = self.state_dict()
        checkpoint = torch.load(pretrained, map_location="cpu")
        loaded_dict = checkpoint.get("state_dict", checkpoint)

        new_state_dict = {}
        for key, val in loaded_dict.items():
            new_key = f"backbone.{key}" if f"backbone.{key}" in cur_state_dict else key
            if new_key not in cur_state_dict:
                continue
            if val.shape != cur_state_dict[new_key].shape:
                continue
            new_state_dict[new_key] = val

        self.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        feats = []
        x = self.stem(x)
        x = self.stage1(x)
        feats.append(x)
        x = self.stage2(x)
        feats.append(x)
        x = self.stage3(x)
        feats.append(x)
        x = self.stage4(x)
        feats.append(x)
        return feats


@BACKBONES.register_module()
class StudentCOSNetBEM(nn.Module):
    """Student backbone with 2x3x3 stem and BEM enhancement at all stages."""

    def __init__(
        self,
        in_chans=3,
        depths=[2, 2, 6, 2],
        dim=48,
        expan_ratio=4,
        num_stages=4,
        s_kernel_size=[5, 5, 3, 3],
        drop_path_rate=0.1,
        layer_scale_init_value=1e-6,
        **kwargs,
    ):
        super().__init__()

        self.num_stages = num_stages
        self.dims = [dim * (2 ** i) for i in range(self.num_stages)]

        self.downsample_layers = nn.ModuleList()
        # Replace 5x5/stride4 stem with two 3x3/stride2 convs.
        stem = nn.Sequential(
            nn.Conv2d(in_chans, self.dims[0], kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(self.dims[0], self.dims[0], kernel_size=3, stride=2, padding=1),
        )
        self.downsample_layers.append(stem)

        for i in range(self.num_stages - 1):
            self.downsample_layers.append(
                nn.Conv2d(self.dims[i], self.dims[i + 1], kernel_size=3, stride=2, padding=1)
            )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stages):
            blocks = []
            for j in range(depths[i]):
                blocks.append(
                    FSB(
                        dim=self.dims[i],
                        s_kernel_size=s_kernel_size[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        expan_ratio=expan_ratio,
                    )
                )
            self.stages.append(nn.Sequential(*blocks))
            cur += depths[i]

        # BEM at every stage + boundary logit heads for dedicated boundary loss.
        self.bem_layers = nn.ModuleList([BEM(self.dims[i]) for i in range(self.num_stages)])
        self.boundary_heads = nn.ModuleList([nn.Conv2d(self.dims[i], 1, kernel_size=1) for i in range(self.num_stages)])
        self.latest_boundary_logits = None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm, nn.BatchNorm2d)):
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return

        cur_state_dict = self.state_dict()
        checkpoint = torch.load(pretrained, map_location="cpu")
        loaded_dict = checkpoint.get("state_dict", checkpoint)

        new_state_dict = {}
        for key, val in loaded_dict.items():
            new_key = f"backbone.{key}" if f"backbone.{key}" in cur_state_dict else key
            if new_key not in cur_state_dict:
                continue
            if val.shape != cur_state_dict[new_key].shape:
                continue
            new_state_dict[new_key] = val

        self.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        feats = []
        boundary_logits = []
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            bem_x = self.bem_layers[i](x)
            boundary_logits.append(self.boundary_heads[i](bem_x))
            x = x + bem_x
            feats.append(x)

        self.latest_boundary_logits = boundary_logits
        return feats
