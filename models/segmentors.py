from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors import EncoderDecoder
from mmseg.ops import resize

from .losses import CombinedLoss


@SEGMENTORS.register_module()
class COSNetEncoderDecoder(EncoderDecoder):
    """EncoderDecoder variant that adds CombinedLoss with backbone boundary outputs."""

    def __init__(self, *args, use_combined_loss=True, combined_loss_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_combined_loss = use_combined_loss

        cfg = dict(combined_loss_cfg or {})
        cfg.setdefault('num_classes', self.decode_head.num_classes)
        self.combined_loss = CombinedLoss(**cfg)

    def forward_train(self, img, img_metas, gt_semantic_seg):
        if not self.use_combined_loss:
            return super().forward_train(img, img_metas, gt_semantic_seg)

        x = self.extract_feat(img)
        losses = dict()

        # Decode logits and align with GT resolution for CE inside CombinedLoss.
        seg_logits = self.decode_head.forward(x)
        if seg_logits.shape[2:] != gt_semantic_seg.shape[2:]:
            seg_logits = resize(
                input=seg_logits,
                size=gt_semantic_seg.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False,
            )

        seg_labels = gt_semantic_seg.squeeze(1).long()

        boundary_logits = getattr(self.backbone, 'boundary_attn_logits', None)
        attn_logits = getattr(self.backbone, 'boundary_attn_logits', None)

        total_loss, loss_dict = self.combined_loss(
            seg_logits=seg_logits,
            seg_labels=seg_labels,
            boundary_logits=boundary_logits,
            attn_logits=attn_logits,
        )

        losses['loss_decode'] = total_loss
        losses['decode_ce'] = total_loss.new_tensor(loss_dict.get('ce', 0.0)).detach()
        losses['decode_boundary'] = total_loss.new_tensor(loss_dict.get('boundary', 0.0)).detach()
        losses['decode_scale_reg'] = total_loss.new_tensor(loss_dict.get('scale_reg', 0.0)).detach()

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses
