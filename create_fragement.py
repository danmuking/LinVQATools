"""
    实现fragment数据预处理
"""
import os
from multiprocessing import freeze_support

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
# import cv2
# import numpy as np
from tqdm import tqdm

from data.default_dataset import DefaultDataset
from data.resize_dataset import ResizeDataset


def makedir(path: str):
    dir_path = os.path.dirname(path)
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
    train_dataset = ResizeDataset(anno_reader='ODVVQAReader',
                             anno_root=r'G:/code/LinVQATools/data/odv_vqa/',
                             norm=False,
                             split_file=r'G:\code\LinVQATools\data\odv_vqa\tr_te_VQA_ODV.txt',
                             frame_sampler=frame_sampler, spatial_sampler=spatial_sampler, phase='test')
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
        # print(video_path)
        video_path.insert(2, 'resize')
        video_path[0] = "G:\\"
        video_path[1] = ""
        video_path = os.path.join(*video_path)[:-4]+'\\'
        # video_path = os.path.join('D:/code/LinVQATools/data/odv_vqa/',video_path)
        makedir(video_path)
        video = data['inputs']
        video = torch.stack(video,0).permute(1,3,0,2,4,5)
        print(video.shape)
        video = video[0][0]
        print(video.shape)
        video = video.permute(0,2,3,1).numpy().astype(np.uint8)
        print(video_path)
        print(data['name'])
        print(video.shape)
        for i in range(video.shape[0]):
            img = video[i]
            img_path = os.path.join(video_path,"{}.png".format(i))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, img)  # 存入快照
        # break

        # torch.save(video,video_path)
        # print(data)
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # # 设置视频帧频
        # fps = 10
        # # 设置视频大小
        # size = video.shape[-2], video.shape[-1]
        #
        # out = cv2.VideoWriter(video_path, fourcc, fps, size)
        # for i in range(video.shape[1]):
        #     fra = video[:, i, :, :]
        #     fra = fra.permute(1, 2, 0)
        #     fra = fra.numpy().astype(np.uint8)
        #     fra = cv2.cvtColor(fra, cv2.COLOR_RGB2BGR)
        #     out.write(fra)
        # out.release()
        index = index + 1



