import os

import torch
from tqdm import tqdm

from data.default_dataset import SingleBranchDataset
from models.video_mae_vqa import VideoMAEVQAWrapper

os.chdir('../')
video_loader = dict(
    name='FragmentLoader',
    prefix='4frame',
    argument=[
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
)

dataset = SingleBranchDataset(video_loader=video_loader,
                              anno_root='./data/odv_vqa',
                              anno_reader='ODVVQAReader',
                              split_file='./data/odv_vqa/tr_te_VQA_ODV.txt',
                              phase='train',
                              norm=True)
model = VideoMAEVQAWrapper(model_type="s",mask_ratio=0.75)
# model.load_state_dict(torch.load('/data/ly/code/LinVQATools/work_dir/video_mae_vqa/12151416 vit random_cell_mask_75 mae last6 4clip/best_SROCC_epoch_3.pth')['state_dict'])


# 记录三个字段
name_List = []
gt_label_List = []
pre_List = []

for data in tqdm(dataset):
    inputs = data['inputs'].unsqueeze(0)
    name = data['name']
    gt_label = data['gt_label']
    with torch.no_grad():
        y = model(inputs=inputs, gt_label=torch.rand((2)),mode='predict')
    pre = y[0]
    name_List.append(name)
    gt_label_List.append(gt_label)
    pre_List.append(pre.detach().numpy()[0])


dataset = SingleBranchDataset(video_loader=video_loader,
                              anno_root='./data/odv_vqa',
                              anno_reader='ODVVQAReader',
                              split_file='./data/odv_vqa/tr_te_VQA_ODV.txt',
                              phase='test',
                              norm=True)


for data in tqdm(dataset):
    inputs = data['inputs'].unsqueeze(0)
    name = data['name']
    gt_label = data['gt_label']
    with torch.no_grad():
        y = model(inputs=inputs, gt_label=torch.rand((2)),mode='predict')
    pre = y[0]
    name_List.append(name)
    gt_label_List.append(gt_label)
    pre_List.append(pre.detach().numpy()[0])

# 将列表写入文件
with open('score.txt', 'w') as f:
    for i in range(len(name_List)):
        f.write(name_List[i] + ' ' + str(gt_label_List[i]) + ' ' + str(pre_List[i]) + '\n')
    f.close()
