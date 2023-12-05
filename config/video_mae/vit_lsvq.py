custom_imports = dict(
    imports=['video_mae_vqa', 'lsvq', 'srocc', 'rmse',
             'plcc', 'krcc', 'train_evaluator_hook', 'custom_ema_hook'],
    allow_failed_imports=False)
name = "12051819 vit random_cell_mask_75 mae last6 4clip lsvq"
work_dir = 'work_dir/video_mae_vqa/'+name
visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(project='video mae vqa', name=name)
        ),
    ],
)
model = dict(
    type='VideoMAEVQAWrapper',
    model_type='s',
    mask_ratio=0.75,
    head_dropout=0.1,
    drop_path_rate=0.1
)
epochs = 60
batch_size = 16
num_workers = 6
argument = [
        dict(
            name='FragmentShuffler',
            fragment_size=32,
            frame_cube=4
        ),
]
train_dataloader = dict(
    dataset=dict(
        type='LSVQDataset',
        argument=argument,
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
val_dataloader = dict(
    dataset=dict(
        type='LSVQDataset',
        argument=argument,
        phase='test',
        norm=True,
        clip=4
    ),
    sampler=dict(
        type='DefaultSampler',
        shuffle=False
    ),
    collate_fn=dict(type='default_collate'),
    batch_size=1,
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
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    # accumulative_counts=4,
    paramwise_cfg=dict(
        custom_keys={
            # 'model.backbone': dict(lr_mult=0.1),
            # 'model.decoder': dict(lr_mult=0.1),
        })
)
param_scheduler = [
    # 在 [0, 100) 迭代时使用线性学习率
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=3,
        convert_to_iter_based=True
    ),
    # 在 [100, 900) 迭代时使用余弦学习率
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=3,
        T_max=epochs,
        eta_min=0.00001*0.01,
        convert_to_iter_based=True
    ),
]

val_evaluator = [
    dict(type='SROCC'),
    dict(type='KRCC'),
    dict(type='PLCC'),
    dict(type='RMSE'),
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best='SROCC', rule='greater'))
custom_hooks = [
    # dict(type='TrainEvaluatorHook'),
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
