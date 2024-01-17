import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.video_mae_vqa import VideoMAEVQAWrapper
from data.default_dataset import SingleBranchDataset
os.chdir('../')

model = VideoMAEVQAWrapper(
    model_type='s',
    mask_ratio=0.75,
    head_dropout=0.1,
    drop_path_rate=0.1
)
weight_path = '/data/ly/code/LinVQATools/pretrained_weights/best_SROCC_epoch_316.pth'
weight = torch.load(weight_path,map_location="cpu")
info = model.load_state_dict(weight['state_dict'])
print(info)

num_workers = 6
prefix = '4frame'
argument = [
        dict(
            name='FragmentShuffler',
            fragment_size=32,
            frame_cube=4
        ),
        dict(
            name='PostProcessSampler',
            frame_cube=4,
            num=4
        )
]
val_video_loader = dict(
    name='FragmentLoader',
    prefix=prefix,
    frame_sampler=None,
    spatial_sampler=None,
    argument=argument,
    phase='test',
    use_preprocess=True,
)
dataset=SingleBranchDataset(
    video_loader=val_video_loader,
    anno_root='./data/odv_vqa',
    anno_reader='ODVVQAReader',
    split_file='./data/odv_vqa/tr_te_VQA_ODV.txt',
    phase='test',
    norm=True,
    clip=4
)
val_dataloader = DataLoader(batch_size=1,
                            shuffle=False,
                            dataset=dataset)

gt = []
pr = []
model = model.cuda().eval()
with torch.no_grad():
    for i,item in enumerate(val_dataloader):
        gt.append(item['gt_label'])
        print(i)
        inputs = item["inputs"].cuda()
        gt_label = item['gt_label'].cuda()
        y = model(inputs=inputs, gt_label=gt_label,mode='predict')
        pr.append(y[0])
        # print(i)