import os
from unittest import TestCase

import cv2
import numpy as np
import torch

from data.default_dataset import SingleBranchDataset
from data.shuffler import MixShuffler


class TestSingleBranchDataset(TestCase):
    def test(self):
        os.chdir('../../')
        video_loader = dict(
            name='FragmentLoader',
            prefix='fragment',
            argument=[
                # dict(
                #     name='FragmentShuffler',
                #     fragment_size=112,
                #     frame_cube=4
                # ),
                dict(
                    name='MixShuffler',
                    fragment_size=32,
                ),
                dict(
                    name='PostProcessSampler',
                    num=4,
                    frame_cube=4
                )
            ]
        )
        dataset = SingleBranchDataset(video_loader=video_loader, norm=False)
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

    def test_mix(self):
        shuffler = MixShuffler(32)
        video = [x for x in range(224*224)]
        video = torch.tensor(video).reshape(1,1,224,224)
        shuffler(video)
