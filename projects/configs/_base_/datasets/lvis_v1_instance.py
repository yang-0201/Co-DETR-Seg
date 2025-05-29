# dataset settings
_base_ = 'coco_instance.py'
dataset_type = 'LVISV1Dataset'
data_root = '/home/data/Co-DETR/train/'
train_pipeline = [
    dict(type='CopyPaste', max_num_pasted=100),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'train_annotations.json',
            img_prefix=data_root+'images/'),
            pipeline = train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'train_annotations.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'train_annotations.json',
        img_prefix=data_root + 'images/'))
evaluation = dict(metric=['bbox', 'segm'])
