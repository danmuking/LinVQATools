custom_imports = dict(
    imports=['faster_vqa', 'default_dataset', 'srocc', 'rmse',
             'plcc', 'krcc', 'train_evaluator_hook', 'custom_ema_hook'],
    allow_failed_imports=False)
work_dir = 'work_dir/faster_vqa/vit_patch32_fragment32'
visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(project='faster vqa消融', name='vit patch32 fragment32')
        ),
    ],
)
model = dict(
    type='FasterVQA',
    backbone='vit',
    base_x_size=(16, 224, 224),
    window_size=(8, 7, 7),
    vqa_head=dict(name='VQAHead', in_channels=384, drop_rate=0.5,fc_in=8*7*7),
    # vqa_head=dict(name='FcHead', in_channels=384, drop_rate=0.5),
    load_path="./pretrained_weights/vit_s_k710_dl_from_giant.pth"
)
epochs = 600
batch_size = 16
num_workers = 16
base_lr = 0.001
prefix = 'temp/fragment'
argument = [
    dict(
        name='FragmentShuffler',
        fragment_size=32,
    ),
    dict(
        name='PostProcessSampler',
        num=2
    )
]
train_video_loader = dict(
    name='FragmentLoader',
    prefix=prefix,
    frame_sampler=None,
    spatial_sampler=None,
    argument=argument,
    phase='train',
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
    prefix=prefix,
    frame_sampler=None,
    spatial_sampler=None,
    argument=argument,
    phase='test',
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
    optimizer=dict(type='AdamW', lr=base_lr * batch_size / 256, weight_decay=0.05),
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