import os
from unittest import TestCase

import cv2
import numpy as np

from data.live_vqc import Live_VQCDataset


class TestLive_VQCDataset(TestCase):
    def test(self):
        os.chdir('../../')
        argument = [
            dict(
                name='FragmentShuffler',
                fragment_size=32,
                frame_cube=4
            ),
        ]
        dataset = Live_VQCDataset(norm=False, argument=argument, clip=4)
        data = dataset[0]
        # video = torch.from_numpy(np.load("temp.npy"))
        video = data['inputs']
        print(video.shape)
        video = video[1]
        # pos_embed = data['pos_embed']
        # print(pos_embed.shape)
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
