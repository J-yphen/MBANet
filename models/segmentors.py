from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors import EncoderDecoder
from mmseg.ops import resize

from .losses import CombinedLoss


@SEGMENTORS.register_module()
class COSNetEncoderDecoder(EncoderDecoder):
    """EncoderDecoder variant that adds CombinedLoss with backbone boundary outputs."""

    def __init__(self,
                 *args,
                 use_combined_loss=True,
                 combined_loss_cfg=None,
                 adaptive_boundary_cfg=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_combined_loss = use_combined_loss

        cfg = dict(combined_loss_cfg or {})
        cfg.setdefault('num_classes', self.decode_head.num_classes)
        self.combined_loss = CombinedLoss(**cfg)

        default_adaptive_cfg = dict(
            enabled=False,
            min_lambda_bound=self.combined_loss.lambda_bound,
            max_lambda_bound=self.combined_loss.lambda_bound,
            warmup_iters=8000,
            plateau_window=300,
            plateau_delta=5e-4,
            plateau_boost=0.02,
            max_plateau_boost=0.20,
            decay_on_improve=0.5,
        )
        if adaptive_boundary_cfg is not None:
            default_adaptive_cfg.update(adaptive_boundary_cfg)
        self.adaptive_boundary_cfg = default_adaptive_cfg

        self._train_step = 0
        self._ce_history = []
        self._plateau_boost = 0.0

    def _compute_dynamic_lambda_bound(self):
        cfg = self.adaptive_boundary_cfg
        base_lambda = float(self.combined_loss.lambda_bound)
        if not cfg.get('enabled', False):
            return base_lambda

        min_lambda = float(cfg.get('min_lambda_bound', base_lambda))
        max_lambda = float(cfg.get('max_lambda_bound', base_lambda))
        if max_lambda < min_lambda:
            max_lambda = min_lambda

        warmup_iters = max(1, int(cfg.get('warmup_iters', 1)))
        warmup_ratio = min(1.0, float(self._train_step) / float(warmup_iters))
        warmup_lambda = min_lambda + (max_lambda - min_lambda) * warmup_ratio

        max_plateau_boost = max(0.0, float(cfg.get('max_plateau_boost', 0.0)))
        dynamic_lambda = warmup_lambda + self._plateau_boost
        dynamic_lambda = min(dynamic_lambda, max_lambda + max_plateau_boost)
        dynamic_lambda = max(dynamic_lambda, min_lambda)
        return dynamic_lambda

    def _update_plateau_boost(self, ce_value):
        cfg = self.adaptive_boundary_cfg
        if not cfg.get('enabled', False):
            return

        window = max(10, int(cfg.get('plateau_window', 300)))
        self._ce_history.append(float(ce_value))
        if len(self._ce_history) > 2 * window:
            self._ce_history = self._ce_history[-2 * window:]

        if len(self._ce_history) < 2 * window:
            return
        if self._train_step % window != 0:
            return

        prev_mean = sum(self._ce_history[:window]) / float(window)
        curr_mean = sum(self._ce_history[window:]) / float(window)
        improve = prev_mean - curr_mean

        plateau_delta = float(cfg.get('plateau_delta', 5e-4))
        plateau_step = max(0.0, float(cfg.get('plateau_boost', 0.0)))
        max_boost = max(0.0, float(cfg.get('max_plateau_boost', 0.0)))
        decay_on_improve = max(0.0, float(cfg.get('decay_on_improve', 0.0)))

        if improve < plateau_delta:
            self._plateau_boost = min(max_boost, self._plateau_boost + plateau_step)
        else:
            self._plateau_boost = max(0.0, self._plateau_boost - plateau_step * decay_on_improve)

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
        dynamic_lambda_bound = self._compute_dynamic_lambda_bound()

        total_loss, loss_dict = self.combined_loss(
            seg_logits=seg_logits,
            seg_labels=seg_labels,
            boundary_logits=boundary_logits,
            attn_logits=attn_logits,
            lambda_bound=dynamic_lambda_bound,
        )

        self._update_plateau_boost(loss_dict.get('ce', 0.0))
        self._train_step += 1

        losses['loss_decode'] = total_loss
        losses['decode_ce'] = total_loss.new_tensor(loss_dict.get('ce', 0.0)).detach()
        losses['decode_boundary'] = total_loss.new_tensor(loss_dict.get('boundary', 0.0)).detach()
        losses['decode_scale_reg'] = total_loss.new_tensor(loss_dict.get('scale_reg', 0.0)).detach()
        losses['decode_lambda_bound'] = total_loss.new_tensor(
            loss_dict.get('lambda_bound', self.combined_loss.lambda_bound)
        ).detach()
        losses['decode_boundary_boost'] = total_loss.new_tensor(self._plateau_boost).detach()

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses
