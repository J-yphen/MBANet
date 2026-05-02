_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/zero_waste.py',
    '../_base_/default_runtime.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
# model settings
model = dict(
    type='COSNetEncoderDecoder',
    pretrained='',  # pretrain (imagenet) weight path 
    combined_loss_cfg=dict(
        lambda_bound=0.4,
        lambda_scale=0.05,
        dilation_r=3,
        pos_weight=5.0,
        ignore_index=255,
    ),
    adaptive_boundary_cfg=dict(
        enabled=True,
        min_lambda_bound=0.20,
        max_lambda_bound=0.50,
        warmup_iters=12000,
        plateau_window=400,
        plateau_delta=5e-4,
        plateau_boost=0.01,
        max_plateau_boost=0.12,
        decay_on_improve=0.5,
    ),
    backbone=dict(
        type='COSNet',
        depths=[3, 3, 12, 3],
        mba_pool_scales=(2, 4, 8),
        mba_reduction=4,
        style='pytorch'),
    decode_head=dict(num_classes=5,
                     in_channels=[72, 72*2, 72*4, 72*8],
                     channels=256,
                     in_index=[0, 1, 2, 3],
                     norm_cfg=norm_cfg),
    auxiliary_head=dict(num_classes=5,
                        in_channels=72*4,
                        in_index=2,         #in_index=4,
                        norm_cfg=norm_cfg)
    )



gpu_multiples = 1  # we used 1 gpu
# optimizer
optimizer = dict(type='AdamW', lr=0.00009*gpu_multiples, betas=(0.9, 0.999), weight_decay=0.001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', warmup='linear', warmup_iters=1500,
                 warmup_ratio=1e-6, power=0.95, min_lr=1e-7, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000//gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=4000//gpu_multiples)
evaluation = dict(interval=4000//gpu_multiples, metric='mIoU', save_best='mIoU')