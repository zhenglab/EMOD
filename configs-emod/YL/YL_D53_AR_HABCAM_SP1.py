_base_ = [
    '../_base_/models/yolov3_d53.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        type='DarknetEMOD',
        ar=dict(ratio=1. / 4.),
        stage_with_ar=(True, True, True, True, True)
    ),
    bbox_head=dict(num_classes=11))
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/underwater/'
classes=('Gastropoda', 'Osteroida', 'Cephalopoda', 'Decapoda', 'Aplousobranchia', 'NotPleuronectiformes', 'Perciformes', 'Fish', 'Rajiformes', 'NonLiving', 'Pleuronectiformes')
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=[(320, 320), (608, 608)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/habcam_seq0_training_1.mscoco.json',
        img_prefix=data_root + 'habcam_seq0/',
        pipeline=train_pipeline), 
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/habcam_seq0_validation_1.mscoco.json',
        img_prefix=data_root + 'habcam_seq0/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/habcam_seq0_validation_1.mscoco.json',
        img_prefix=data_root + 'habcam_seq0/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2),_delete_=True)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.1,
    step=[218, 246])
runner = dict(type='EpochBasedRunner', max_epochs=300)
