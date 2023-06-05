"""
    实现fragment数据预处理
"""
import os

import cv2
import numpy as np
from torch.utils.data import DataLoader
# import cv2
# import numpy as np
from tqdm import tqdm

from data.default_dataset import DefaultDataset


def makedir(path: str):
    dir_path = os.path.dirname(path)
    if (os.path.exists(dir_path)):
        pass
    else:
        os.makedirs(dir_path)


os.chdir('/')
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
dataset = DefaultDataset(anno_reader='ODVVQAReader',
                         split_file='/home/ly/code/LinVQATools/data/odv_vqa/tr_te_VQA_ODV.txt',
                         frame_sampler=frame_sampler, spatial_sampler=spatial_sampler, phase='test')
dataloader = DataLoader(dataset, batch_size=1, num_workers=6, shuffle=False)
for j in range(80):
    index = 0
    for item in tqdm(dataloader):
        data = item
        video_info = dataset.data[index]
        video_path = video_info["video_path"]
        video_path = video_path.split('/')
        video_path.insert(3, 'fragment')
        video_path.insert(4, '{}'.format(j))
        video_path = os.path.join('/', *video_path)
        # video_path = os.path.join('D:/code/LinVQATools/data/odv_vqa/',video_path)
        makedir(video_path)
        video = data['inputs']
        video = video[0]
        # print(data)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 设置视频帧频
        fps = 30
        # 设置视频大小
        size = video.shape[-2], video.shape[-1]
        print(video_path)
        print(data['name'])
        out = cv2.VideoWriter(video_path, fourcc, fps, size)
        for i in range(video.shape[1]):
            fra = video[:, i, :, :]
            fra = fra.permute(1, 2, 0)
            fra = fra.numpy().astype(np.uint8)
            fra = cv2.cvtColor(fra, cv2.COLOR_RGB2BGR)
            out.write(fra)
        out.release()
        index = index + 1
