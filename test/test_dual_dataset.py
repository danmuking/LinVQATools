import os
from unittest import TestCase

import cv2
import numpy as np

from data.dual_dataset import DualDataset


class TestDualDataset(TestCase):
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
            name='SphereSpatialFragmentSampler',
            fragments_h=7,
            fragments_w=7,
            fsize_h=32,
            fsize_w=32,
            aligned=8,
        )
        shuffler = dict(
            name='BaseShuffler',
        )
        post_sampler = dict(
            name='PostProcessSampler',
            num=2
        )
        prefix = 'temp/fragment'
        loader1 = dict(
            name='FragmentLoader',
            frame_sampler=frame_sampler,
            spatial_sampler=spatial_sampler,
            shuffler=shuffler,
            post_sampler=post_sampler,
            norm=False,
            prefix=prefix,
        )
        loader2 = dict(
            name='RandomImgLoader',
            shuffler=shuffler,
            norm=False,
            prefix='resize'
        )
        split_file = './data/odv_vqa/tr_te_VQA_ODV.txt'
        dataset = DualDataset(anno_reader='ODVVQAReader', split_file=split_file,
                              loader1=loader1,loader2=loader2,phase='test')
        data = dataset[0]
        video = data['inputs'][1]
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
