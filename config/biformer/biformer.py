custom_imports = dict(
    imports=['biformer_arc', 'default_dataset', 'srocc', 'rmse',
             'plcc', 'krcc', 'train_evaluator_hook'],
    allow_failed_imports=False)
work_dir = 'work_dir/biformer/base'
visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(project='faster vqa消融', name='10032355 biformer')
        ),
    ],
)
model = dict(
    type='BiFormerArc',
    load_path="./pretrained_weights/biformer_tiny_best.pth",
    vqa_head=dict(name='VQAHead', in_channels=512,fc_in=4*7*7)
)
epochs = 600
batch_size = 4
num_workers = 6
prefix = 'fragment'
argument = [
    dict(
        name='FragmentShuffler',
        fragment_size=32,
    ),
    dict(
        name='PostProcessSampler',
        num=1
    )
]
train_video_loader = dict(
    name='FragmentLoader',
    frame_sampler=None,
    spatial_sampler=None,
    argument=argument,
    phase='train',
    prefix=prefix,
    use_preprocess=True,
)
train_dataloader = dict(
    dataset=dict(
        type='SingleBranchDataset',
        video_loader=train_video_loader,
        anno_root='./data/odv_vqa',
        anno_reader='ODVVQAReader',
        split_file='./data/odv_vqa/tr_te_VQA_ODV.txt',
        phase='train',
        norm=True
    ),
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    collate_fn=dict(type='default_collate'),
    batch_size=batch_size,
    pin_memory=True,
    num_workers=num_workers)
val_video_loader = dict(
    name='FragmentLoader',
    frame_sampler=None,
    spatial_sampler=None,
    argument=argument,
    phase='test',
    prefix=prefix,
    use_preprocess=True,
)
val_dataloader = dict(
    dataset=dict(
        type='SingleBranchDataset',
        video_loader=val_video_loader,
        anno_root='./data/odv_vqa',
        anno_reader='ODVVQAReader',
        split_file='./data/odv_vqa/tr_te_VQA_ODV.txt',
        phase='test',
        norm=True
    ),
    sampler=dict(
        type='DefaultSampler',
        shuffle=False
    ),
    collate_fn=dict(type='default_collate'),
    batch_size=batch_size,
    pin_memory=True,
    num_workers=num_workers)
train_cfg = dict(
    by_epoch=True,
    max_epochs=epochs,
    val_begin=1,
    val_interval=1)
val_cfg = dict()
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.05),
    # accumulative_counts=4,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
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
        T_max=epochs,
        # eta_min=0.00002,
        convert_to_iter_based=True
    )
]
val_evaluator = [
    dict(type='SROCC'),
    dict(type='KRCC'),
    dict(type='PLCC'),
    dict(type='RMSE'),
]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=10, save_best='SROCC', rule='greater'))
custom_hooks = [
    # dict(type='TrainEvaluatorHook'),
    # dict(type='CustomEMAHook',momentum=0.01)
    # dict(type='EmptyCacheHook', after_epoch=True)
]
launcher = 'pytorch'
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
