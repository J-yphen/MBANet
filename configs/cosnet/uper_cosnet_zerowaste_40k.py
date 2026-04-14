_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/zero_waste.py',
    '../_base_/default_runtime.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
# model settings
model = dict(
    type='EncoderDecoder',
    pretrained='/home/mtech2025/Documents/project_CosNet/sdc1/cosnet/pretrain/model_best_fixed.pth',
    backbone=dict(
        type='COSNet',
        depths=[3, 3, 12, 3],
        style='pytorch'),
    decode_head=dict(num_classes=5,
                     in_channels=[72, 72*2, 72*4, 72*8],
                     channels=256,
                     in_index=[0, 1, 2, 3],
                     norm_cfg=norm_cfg),
    auxiliary_head=[
        dict(
            type='FCNHead',
            num_classes=5,
            in_channels=72,
            in_index=0,
            channels=64,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.20)),
        dict(
            type='FCNHead',
            num_classes=5,
            in_channels=72*2,
            in_index=1,
            channels=96,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.25)),
        dict(
            type='FCNHead',
            num_classes=5,
            in_channels=72*4,
            in_index=2,
            channels=128,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.30)),
    ]
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
