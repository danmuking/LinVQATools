"""
    实现fragment数据预处理
"""
import os
from multiprocessing import freeze_support

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
# import cv2
# import numpy as np
from tqdm import tqdm

from data.resize_dataset import ResizeDataset
from data.spatio_dataset import SpatioDataset


def makedir(path: str):
    # dir_path = os.path.dirname(path)
    dir_path = path
    if (os.path.exists(dir_path)):
        pass
    else:
        os.makedirs(dir_path)


if __name__ == '__main__':

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
    freeze_support()
    train_dataset = SpatioDataset(anno_reader='ODVVQAReader',
                             anno_root=r'/home/ly/data/code/LinVQATools/data/odv_vqa',
                             norm=False,
                             split_file=r'/home/ly/data/code/LinVQATools/data/odv_vqa/tr_te_VQA_ODV.txt',
                             frame_sampler=frame_sampler, spatial_sampler=spatial_sampler, phase='train')
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=False)
    # test_dataset = DefaultDataset(anno_reader='ODVVQAReader',
    #                          anno_root=r'G:/code/LinVQATools/data/odv_vqa/',
    #                          norm=False,
    #                          split_file=r'G:\code\LinVQATools\data\odv_vqa\tr_te_VQA_ODV.txt',
    #                          frame_sampler=frame_sampler, spatial_sampler=spatial_sampler, phase='test')
    # test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)
    index = 0
    for item in tqdm(train_dataloader):
        data = item
        video_info = train_dataset.data[index]
        video_path = video_info["video_path"]
        video_path = video_path.split('/')
        video_path.insert(2, 'spatio')
        video_path[0] = "/data/ly/"
        video_path[1] = ""
        video_path[3] = ""
        video_path = os.path.join(*video_path)[:-4]
        # video_path = os.path.join('D:/code/LinVQATools/data/odv_vqa/',video_path)
        # print(video_path)
        makedir(video_path)
        video = data['inputs']
        # print(len(video))
        for i in range(len(video)):
            # 将 Tensor 转为 PIL 图像
            to_pil = ToPILImage()  # 使用 torchvision.transforms.ToPILImage
            image = to_pil(video[i][0])
            img_path = os.path.join(video_path,"{}.png".format(i))
            image.save(img_path)
        index = index + 1
        # break



