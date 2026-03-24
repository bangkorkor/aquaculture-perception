# Shared UATD experiment recipe for fair cross-library comparison
# pretrained = yes
# augmentation = explicit and limited
# two-stage models = Faster R-CNN / Dynamic R-CNN / similar FPN-based detectors

classes = (
    'human-body',
    'ball',
    'circle-cage',
    'square-cage',
    'tyre',
    'metal-bucket',
    'cube',
    'cylinder',
    'plane',
    'rov',
)
num_classes = len(classes)
metainfo = dict(classes=classes)

dataset_type = 'CocoDataset'
data_root = '../data-processing/sonar/UATD/processed/'
backend_args = None

# Explicit, limited augmentation recipe:
# - resize to 640
# - horizontal flip only
# - no mosaic, mixup, hsv, translate, scale, perspective, etc.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]

# Kept parallel to your RUOD two-stage setup.
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.02,
        momentum=0.9,
        weight_decay=1e-4,
    ),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=80,
        by_epoch=True,
        milestones=[53, 73],
        gamma=0.1,
    ),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

randomness = dict(seed=0, deterministic=False)

auto_scale_lr = dict(enable=False, base_batch_size=16)