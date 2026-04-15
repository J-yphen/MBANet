import torch
import torch.nn.functional as F

from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors import EncoderDecoder


@SEGMENTORS.register_module()
class BoundaryEncoderDecoder(EncoderDecoder):
    def __init__(self,
                 boundary_loss_weight=0.1,
                 boundary_dilation=2,
                 boundary_stage_idx=2,
                 boundary_ignore_index=255,
                 **kwargs):
        super().__init__(**kwargs)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.boundary_dilation = int(boundary_dilation)
        self.boundary_stage_idx = int(boundary_stage_idx)
        self.boundary_ignore_index = int(boundary_ignore_index)

    def _compute_boundary_target(self, gt_semantic_seg):
        # gt_semantic_seg: [B, 1, H, W]
        labels = gt_semantic_seg.long()
        valid = labels != self.boundary_ignore_index

        edge = torch.zeros_like(labels, dtype=torch.bool)

        diff_h = (labels[:, :, 1:, :] != labels[:, :, :-1, :]) & valid[:, :, 1:, :] & valid[:, :, :-1, :]
        edge[:, :, 1:, :] |= diff_h
        edge[:, :, :-1, :] |= diff_h

        diff_w = (labels[:, :, :, 1:] != labels[:, :, :, :-1]) & valid[:, :, :, 1:] & valid[:, :, :, :-1]
        edge[:, :, :, 1:] |= diff_w
        edge[:, :, :, :-1] |= diff_w

        edge = edge.float()
        if self.boundary_dilation > 0:
            k = self.boundary_dilation * 2 + 1
            edge = F.max_pool2d(edge, kernel_size=k, stride=1, padding=self.boundary_dilation)
            edge = (edge > 0).float()

        return edge

    def _boundary_aux_loss(self, gt_semantic_seg):
        backbone = self.backbone
        target = self._compute_boundary_target(gt_semantic_seg)

        # Preferred path for current COSNet: use stage boundary logits.
        logits_store = getattr(backbone, 'latest_boundary_logits', None)
        if logits_store is not None and self.boundary_stage_idx < len(logits_store):
            stage_logits = logits_store[self.boundary_stage_idx]
            if stage_logits is not None:
                pred = torch.nan_to_num(stage_logits, nan=0.0, posinf=30.0, neginf=-30.0)
                resized_target = F.interpolate(target, size=pred.shape[2:], mode='nearest')
                loss = F.binary_cross_entropy_with_logits(pred, resized_target)
                return loss * self.boundary_loss_weight

        # Backward-compatible fallback for older multi-residual boundary stores.
        residual_store = getattr(backbone, 'latest_boundary_residuals', None)
        if residual_store is None:
            return None

        if self.boundary_stage_idx >= len(residual_store):
            return None

        stage_data = residual_store[self.boundary_stage_idx]
        if stage_data is None:
            return None

        residual_maps = stage_data.get('residual_maps', None)
        if residual_maps is None or len(residual_maps) == 0:
            return None

        # Concatenate per-scale residual maps along channel dimension.
        pred = torch.cat(residual_maps, dim=1)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=30.0, neginf=-30.0)
        target = F.interpolate(target, size=pred.shape[2:], mode='nearest')
        target = target.expand(-1, pred.shape[1], -1, -1)
        target = target.clamp(0.0, 1.0)

        # Use logits formulation for better numerical stability under mixed precision.
        loss = F.binary_cross_entropy_with_logits(pred, target)
        return loss * self.boundary_loss_weight

    def forward_train(self, img, img_metas, gt_semantic_seg):
        losses = super().forward_train(img, img_metas, gt_semantic_seg)

        boundary_loss = self._boundary_aux_loss(gt_semantic_seg)
        if boundary_loss is not None:
            losses['loss_boundary'] = boundary_loss

        return losses
