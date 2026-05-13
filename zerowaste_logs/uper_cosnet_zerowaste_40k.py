checkpoint_config = dict(by_epoch=False, interval=4000)
crop_size = (
    512,
    512,
)
cudnn_benchmark = True
data = dict(
    samples_per_gpu=8,
    test=dict(
        ann_dir='test/sem_seg',
        data_root=
        '/home/mtech2025/Documents/sdc1/zerowaste-f-final/splits_final_deblurred',
        img_dir='test/data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                img_scale=(
                    2048,
                    512,
                ),
                transforms=[
                    dict(keep_ratio=True, size_divisor=32, type='AlignResize'),
                    dict(type='RandomFlip'),
                    dict(
                        mean=[
                            123.675,
                            116.28,
                            103.53,
                        ],
                        std=[
                            58.395,
                            57.12,
                            57.375,
                        ],
                        to_rgb=True,
                        type='Normalize'),
                    dict(keys=[
                        'img',
                    ], type='ImageToTensor'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        type='ZeroWasteDataset'),
    train=dict(
        dataset=dict(
            ann_dir='train/sem_seg',
            data_root=
            '/home/mtech2025/Documents/sdc1/zerowaste-f-final/splits_final_deblurred',
            img_dir='train/data',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(reduce_zero_label=False, type='LoadAnnotations'),
                dict(
                    img_scale=(
                        2048,
                        512,
                    ),
                    ratio_range=(
                        0.5,
                        2.0,
                    ),
                    type='Resize'),
                dict(
                    cat_max_ratio=0.75,
                    crop_size=(
                        512,
                        512,
                    ),
                    type='RandomCrop'),
                dict(prob=0.5, type='RandomFlip'),
                dict(type='PhotoMetricDistortion'),
                dict(
                    mean=[
                        123.675,
                        116.28,
                        103.53,
                    ],
                    std=[
                        58.395,
                        57.12,
                        57.375,
                    ],
                    to_rgb=True,
                    type='Normalize'),
                dict(
                    pad_val=0, seg_pad_val=255, size=(
                        512,
                        512,
                    ), type='Pad'),
                dict(type='DefaultFormatBundle'),
                dict(keys=[
                    'img',
                    'gt_semantic_seg',
                ], type='Collect'),
            ],
            type='ZeroWasteDataset'),
        times=50,
        type='RepeatDataset'),
    val=dict(
        ann_dir='val/sem_seg',
        data_root=
        '/home/mtech2025/Documents/sdc1/zerowaste-f-final/splits_final_deblurred',
        img_dir='val/data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                img_scale=(
                    2048,
                    512,
                ),
                transforms=[
                    dict(keep_ratio=True, size_divisor=32, type='AlignResize'),
                    dict(type='RandomFlip'),
                    dict(
                        mean=[
                            123.675,
                            116.28,
                            103.53,
                        ],
                        std=[
                            58.395,
                            57.12,
                            57.375,
                        ],
                        to_rgb=True,
                        type='Normalize'),
                    dict(keys=[
                        'img',
                    ], type='ImageToTensor'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        type='ZeroWasteDataset'),
    workers_per_gpu=2)
data_root = '/home/mtech2025/Documents/sdc1/zerowaste-f-final/splits_final_deblurred'
dataset_type = 'ZeroWasteDataset'
device = 'cuda'
dist_params = dict(backend='nccl')
evaluation = dict(interval=4000, metric='mIoU', save_best='mIoU')
gpu_ids = range(0, 1)
gpu_multiples = 1
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
load_from = 'pretrain/cosnet_spectral_waste_iter_40000.pth'
log_config = dict(
    hooks=[
        dict(by_epoch=False, type='TextLoggerHook'),
    ], interval=50)
log_level = 'INFO'
lr_config = dict(
    by_epoch=False,
    min_lr=1e-07,
    policy='poly',
    power=0.95,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06)
model = dict(
    adaptive_boundary_cfg=dict(
        decay_on_improve=0.5,
        enabled=True,
        max_lambda_bound=0.5,
        max_plateau_boost=0.12,
        min_lambda_bound=0.2,
        plateau_boost=0.01,
        plateau_delta=0.0005,
        plateau_window=400,
        warmup_iters=12000),
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=288,
        in_index=2,
        loss_decode=dict(
            avg_non_ignore=True,
            loss_weight=0.5,
            type='CrossEntropyLoss',
            use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=5,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        contract_dilation=True,
        depth=50,
        depths=[
            3,
            3,
            12,
            3,
        ],
        dilations=(
            1,
            1,
            1,
            1,
        ),
        mba_pool_scales=(
            1,
            2,
            4,
        ),
        mba_reduction=4,
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        norm_eval=False,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
            4,
        ),
        strides=(
            1,
            2,
            2,
            2,
        ),
        style='pytorch',
        type='COSNet'),
    combined_loss_cfg=dict(
        dilation_r=3,
        ignore_index=255,
        lambda_bound=0.4,
        lambda_dice=0.5,
        lambda_lovasz=0.75,
        lambda_scale=0.02,
        pos_weight=3.0),
    decode_head=dict(
        align_corners=False,
        channels=256,
        dropout_ratio=0.1,
        in_channels=[
            72,
            144,
            288,
            576,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            avg_non_ignore=True,
            loss_weight=1.0,
            type='CrossEntropyLoss',
            use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=5,
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        type='UPerHead'),
    pretrained='',
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='COSNetEncoderDecoder')
norm_cfg = dict(requires_grad=True, type='BN')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ), lr=9e-05, type='AdamW', weight_decay=0.001)
optimizer_config = dict()
resume_from = None
runner = dict(max_iters=40000, type='IterBasedRunner')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        flip=False,
        img_scale=(
            2048,
            512,
        ),
        transforms=[
            dict(keep_ratio=True, size_divisor=32, type='AlignResize'),
            dict(type='RandomFlip'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(keys=[
                'img',
            ], type='ImageToTensor'),
            dict(keys=[
                'img',
            ], type='Collect'),
        ],
        type='MultiScaleFlipAug'),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(img_scale=(
        2048,
        512,
    ), ratio_range=(
        0.5,
        2.0,
    ), type='Resize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='Normalize'),
    dict(pad_val=0, seg_pad_val=255, size=(
        512,
        512,
    ), type='Pad'),
    dict(type='DefaultFormatBundle'),
    dict(keys=[
        'img',
        'gt_semantic_seg',
    ], type='Collect'),
]
work_dir = 'zerowaste_logs/40k_mba_test_sc/'
workflow = [
    (
        'train',
        1,
    ),
]
