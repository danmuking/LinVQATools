custom_imports = dict(
    imports=['faster_vqa', 'default_dataset', 'srocc', 'rmse', 'plcc', 'krcc', 'train_evaluator_hook'],
    allow_failed_imports=False)
work_dir = 'faster_vqa/basic'
model = dict(
    type='FasterVQA',
    backbone_size='swin_tiny_grpb',
    backbone={"fragments": dict(window_size=(4, 4, 4))},
    backbone_preserve_keys='fragments',
    load_path="./pretrained_weights/FAST_VQA_3D_1_1.pth"
)
train_dataloader = dict(
    dataset=dict(
        type='DefaultDataset',
        prefix='fragment',
        anno_reader='ODVVQAReader',
        split_file='./data/odv_vqa/tr_te_VQA_ODV.txt',
        phase='train',
        frame_sampler=dict(
            name='FragmentSampleFrames',
            fsize_t=32 // 8,
            fragments_t=8,
            clip_len=32,
            frame_interval=2,
            t_frag=8,
            num_clips=1,
        ),
        spatial_sampler=dict(
            name='PlaneSpatialFragmentSampler',
            fragments_h=7,
            fragments_w=7,
            fsize_h=32,
            fsize_w=32,
            aligned=8,
        )
    ),
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    collate_fn=dict(type='default_collate'),
    batch_size=4,
    pin_memory=True,
    num_workers=4)
train_cfg = dict(
    by_epoch=True,
    max_epochs=80,
    val_begin=1,
    val_interval=1)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    accumulative_counts=4,
    paramwise_cfg=dict(
        custom_keys={
            'model.fragments_backbone': dict(lr_mult=1),
        })
)
param_scheduler = [
    # 在 [0, 100) 迭代时使用线性学习率
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=2,
        convert_to_iter_based=True
    ),
    # 在 [100, 900) 迭代时使用余弦学习率
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=2,
        T_max=80,
        convert_to_iter_based=True
    )
]
val_dataloader = dict(
    dataset=dict(
        type='DefaultDataset',
        prefix='fragment',
        anno_reader='ODVVQAReader',
        phase='test',
        split_file='./data/odv_vqa/tr_te_VQA_ODV.txt',
        frame_sampler=dict(
            name='FragmentSampleFrames',
            fsize_t=32 // 8,
            fragments_t=8,
            clip_len=32,
            frame_interval=2,
            t_frag=8,
            num_clips=1,
        ),
        spatial_sampler=dict(
            name='PlaneSpatialFragmentSampler',
            fragments_h=7,
            fragments_w=7,
            fsize_h=32,
            fsize_w=32,
            aligned=8,
        )
    ),
    sampler=dict(
        type='DefaultSampler',
        shuffle=False
    ),
    collate_fn=dict(type='default_collate'),
    batch_size=4,
    pin_memory=True,
    num_workers=4)
val_cfg = dict()
val_evaluator = [
    dict(type='SROCC'),
    dict(type='KRCC'),
    dict(type='PLCC'),
    dict(type='RMSE'),
]

visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(project='VQA', name='basic')
        ),
    ],
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=10, save_best='SROCC', rule='greater'))
custom_hooks = [
    # dict(type='EMAHook'),
    # dict(type='EmptyCacheHook', after_epoch=True)
    dict(type='TrainEvaluatorHook')
]
launcher = 'none'
env_cfg = dict(
    cudnn_benchmark=False,
    backend='nccl',
    mp_cfg=dict(mp_start_method='fork'))
log_level = 'INFO'
load_from = None
resume = False
