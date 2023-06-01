custom_imports = dict(imports=['faster_vqa', 'default_dataset', 'srocc', 'rmse', 'plcc', 'krcc'],
                      allow_failed_imports=False)
work_dir = 'faster_vqa/basic'
model = dict(
    type='FasterVQA',
    backbone_size='swin_tiny_grpb',
    backbone={"fragments": dict(window_size=(4, 4, 4))},
    backbone_preserve_keys='fragments',
    # load_path="./pretrained_weights/FAST_VQA_3D_1_1.pth"
)
val_dataloader = dict(
    dataset=dict(
        type='DefaultDataset',
        anno_reader='ODVVQAReader',
        phase='train',
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
    batch_size=2,
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
            init_kwargs=dict(project='VQA',name='在head后添加全连接层')
        ),
    ],
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1))
custom_hooks = [
    dict(type='EmptyCacheHook', after_epoch=True)]
launcher = 'none'
env_cfg = dict(
    cudnn_benchmark=False,
    backend='nccl',
    mp_cfg=dict(mp_start_method='fork'))
log_level = 'INFO'
load_from = '/home/ly/code/LinVQATools/faster_vqa/basic/epoch_30.pth'
resume = False
