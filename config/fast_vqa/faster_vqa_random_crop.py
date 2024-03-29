custom_imports = dict(
    imports=['faster_vqa', 'default_dataset', 'srocc', 'rmse',
             'plcc', 'krcc', 'train_evaluator_hook', 'custom_ema_hook'],
    allow_failed_imports=False)
work_dir = 'faster_vqa/crop'
model = dict(
    type='FasterVQA',
    backbone='faster_vqa',
    base_x_size=(16, 224, 224),
    vqa_head=dict(in_channels=768),
    load_path="./pretrained_weights/swin_tiny_patch244_window877_kinetics400_1k.pth"
)
train_dataloader = dict(
    dataset=dict(
        type='DefaultDataset',
        prefix='crop',
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
        ),
        shuffler=dict(
            name='BaseShuffler',
        ),
        post_sampler=dict(
            name='PostProcessSampler',
            num=2
        ),
    ),
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    collate_fn=dict(type='default_collate'),
    batch_size=5,
    pin_memory=True,
    num_workers=4)
train_cfg = dict(
    by_epoch=True,
    max_epochs=300,
    val_begin=1,
    val_interval=1)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    # accumulative_counts=4,
    paramwise_cfg=dict(
        custom_keys={
            'model.fragments_backbone': dict(lr_mult=0.1),
        })
)
param_scheduler = [
    # 在 [0, 100) 迭代时使用线性学习率
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True
    ),
    # 在 [100, 900) 迭代时使用余弦学习率
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=10,
        T_max=300,
        # eta_min=0.00002,
        convert_to_iter_based=True
    )
]
val_dataloader = dict(
    dataset=dict(
        type='DefaultDataset',
        anno_reader='ODVVQAReader',
        prefix='crop',
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
        ),
        shuffler=dict(
            name='BaseShuffler',
        ),
        post_sampler=dict(
            name='PostProcessSampler',
            num=2
        ),
    ),
    sampler=dict(
        type='DefaultSampler',
        shuffle=False
    ),
    collate_fn=dict(type='default_collate'),
    batch_size=5,
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
            init_kwargs=dict(project='VQA', name='faster_vqa_crop')
        ),
    ],
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=10, save_best='SROCC', rule='greater'))
custom_hooks = [
    dict(type='TrainEvaluatorHook'),
    # dict(type='CustomEMAHook',momentum=0.01)
    # dict(type='EmptyCacheHook', after_epoch=True)
]
launcher = 'none'
randomness = dict(seed=42)
# randomness = dict(seed=3407)
# randomness = dict(seed=114514)
env_cfg = dict(
    cudnn_benchmark=True,
    backend='nccl',
    mp_cfg=dict(mp_start_method='fork'))
log_level = 'INFO'
load_from = None
resume = False
