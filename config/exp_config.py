# 以下代码存放在 example_config.py 文件中
custom_imports = dict(imports=['renet50','accuracy'], allow_failed_imports=False)
model = dict(type='MMResNet50')
work_dir = 'exp/my_awesome_model'
train_dataloader = dict(
    dataset=dict(type='CIFAR10',
                 root='data/cifar10',
                 train=True,
                 download=True,
                 ),
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    collate_fn=dict(type='default_collate'),
    batch_size=64,
    pin_memory=True,
    num_workers=2)
train_cfg = dict(
    by_epoch=True,
    max_epochs=10,
    val_begin=2,
    val_interval=1)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=dict(type='SGD', lr=0.001, momentum=0.9))
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[4, 8],
    gamma=0.1)
val_dataloader = dict(
    dataset=dict(type='CIFAR10',
                 root='data/cifar10',
                 train=False,
                 download=True,
                 ),
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    collate_fn=dict(type='default_collate'),
    batch_size=64,
    pin_memory=True,
    num_workers=2)
val_cfg = dict()
val_evaluator = dict(type='Accuracy')

visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(project='toy-example')
        ),
    ],
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1))
launcher = 'none'
env_cfg = dict(
    cudnn_benchmark=False,
    backend='nccl',
    mp_cfg=dict(mp_start_method='fork'))
log_level = 'INFO'
load_from = None
resume = False
