_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/zero_waste.py',
    '../_base_/default_runtime.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='BasicStudentNet',
        dims=[32, 64, 128, 256]),
    decode_head=dict(
        num_classes=5,
        in_channels=[32, 64, 128, 256],
        channels=128,
        in_index=[0, 1, 2, 3],
        norm_cfg=norm_cfg),
    auxiliary_head=dict(
        num_classes=5,
        in_channels=128,
        in_index=2,
        channels=64,
        norm_cfg=norm_cfg)
)

gpu_multiples = 1
optimizer = dict(type='AdamW', lr=0.00015 * gpu_multiples, betas=(0.9, 0.999), weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', warmup='linear', warmup_iters=500,
                 warmup_ratio=1e-6, power=0.95, min_lr=1e-7, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=10000 // gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=1000 // gpu_multiples)
evaluation = dict(interval=1000 // gpu_multiples, metric='mIoU', save_best='mIoU')
