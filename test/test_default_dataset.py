import io
import os
from unittest import TestCase

import cv2
import numpy as np
import torch
from decord import VideoReader
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.default_dataset import DefaultDataset


class TestDefaultDataset(TestCase):
    def test_default_dataset(self):
        os.chdir('../')
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
        dataset = DefaultDataset(anno_reader='ODVVQAReader', split_file='./data/odv_vqa/tr_te_VQA_ODV.txt',
                                 frame_sampler=frame_sampler, spatial_sampler=spatial_sampler,prefix='temp/fragment')
        dataloader = DataLoader(dataset, batch_size=6, num_workers=4)
        # for item in tqdm(dataloader):
        #     print(item)
        print(dataset[0]['inputs'].shape)

    def test_save_video(self):
        os.chdir('../')
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
            name='SphereSpatialFragmentSampler',
            fragments_h=7,
            fragments_w=7,
            fsize_h=32,
            fsize_w=32,
            aligned=8,
        )
        dataset = DefaultDataset(anno_reader='ODVVQAReader', split_file='./data/odv_vqa/tr_te_VQA_ODV.txt',
                                 frame_sampler=frame_sampler, spatial_sampler=spatial_sampler,prefix='temp/fragment',norm=False)
        data = dataset[0]
        # video = torch.from_numpy(np.load("temp.npy"))
        video = data['inputs']
        # print(data)
        print(video.shape)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # 设置视频帧频
        fps = 10
        # 设置视频大小
        size = video.shape[-2], video.shape[-1]
        out = cv2.VideoWriter('./out.avi', fourcc, fps, size)
        for i in range(video.shape[1]):
            fra = video[:, i, :, :]
            fra = fra.permute(1, 2, 0)
            fra = fra.numpy().astype(np.uint8)
            fra = cv2.cvtColor(fra, cv2.COLOR_RGB2BGR)
            out.write(fra)
        out.release()

    def test(self):
        os.chdir('../')
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
                                 frame_sampler=frame_sampler, spatial_sampler=spatial_sampler, phase='test',
                                 prefix=None, norm=False)
        dataset.shuffler(video=None)

    def test_load(self):
        os.chdir('../')
        load_torch = torch.load('./out.pt')
        print(load_torch)
        vreader = VideoReader('./out.avi')
        frame_dict = {idx: vreader[idx] for idx in range(len(vreader))}
        imgs = [frame_dict[idx] for idx in range(len(vreader))]
        video = torch.stack(imgs, 0).permute(3, 0, 1, 2)
        print(video)
