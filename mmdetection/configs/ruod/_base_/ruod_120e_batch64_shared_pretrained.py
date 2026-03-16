# Shared RUOD experiment recipe for fair cross-library comparison
# pretrained = yes
# augmentation = explicit and limited

classes = (
    'holothurian',
    'echinus',
    'scallop',
    'starfish',
    'fish',
    'coral',
    'diver',
    'cuttlefish',
    'turtle',
    'jellyfish',
)
num_classes = len(classes)
metainfo = dict(classes=classes)

dataset_type = 'CocoDataset'
data_root = '../data-processing/vision/RUOD/processed/'
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
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# Heavy models like DetectoRS may not fit large per-step batches.
# Use batch_size=1 and accumulate gradients to effective batch size 64.
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
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
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
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
    )
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
    )
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

# Match your Ultralytics-visible settings
# epochs=120, batch=64, optimizer=SGD, lr0=0.01, momentum=0.937,
# weight_decay=0.0005, amp=False, workers=2, seed=0, deterministic=False
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=1e-4,
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}
    ),
)

# Shared simple schedule:
# - 3 epoch warmup
# - linear decay to 1% of initial LR by epoch 120
# This is explicit and avoids silently inheriting MMDetection's 1x schedule.
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        end_factor=1.0,
        begin=0,
        end=3,
        by_epoch=True),
    dict(
        type='LinearMomentum',
        start_factor=0.8 / 0.937,
        end_factor=1.0,
        begin=0,
        end=3,
        by_epoch=True),
    dict(
        type='LinearLR',
        start_factor=1.0,
        end_factor=0.01,
        begin=3,
        end=120,
        by_epoch=True),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=120, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

randomness = dict(seed=0, deterministic=False)

auto_scale_lr = dict(enable=False, base_batch_size=64)


default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=-1,
        save_last=True,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=1,
    )
)