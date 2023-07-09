"""
    实现fragment数据预处理
"""
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.generate_unique import GenerateDataset


def makedir(path: str):
    dir_path = os.path.dirname(path)
    if (os.path.exists(dir_path)):
        pass
    else:
        os.makedirs(dir_path)


frame_sampler = dict(
    name='FragmentSampleFrames',
    fsize_t=32 // 8,
    fragments_t=8,
    clip_len=32,
    frame_interval=2,
    t_frag=8,
    num_clips=1,
)
spatial_sampler = dict(
    name='PlaneSpatialFragmentSampler',
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=8,
)
dataset = GenerateDataset(anno_reader='ODVVQAReader',
                          split_file='/home/ly/code/LinVQATools/data/odv_vqa/tr_te_VQA_ODV.txt',
                          frame_sampler=frame_sampler, spatial_sampler=spatial_sampler, phase='test')
# dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
for item in tqdm(range(259,len(dataset))):
    var = dataset[item]
