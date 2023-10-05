import os
from unittest import TestCase

import cv2
import numpy as np

from data.default_dataset import SingleBranchDataset


class TestSingleBranchDataset(TestCase):
    def test(self):
        os.chdir('../../')
        video_loader = dict(
            name='FragmentLoader',
            prefix='fragment',
            argument=[
                # dict(
                #     name='FragmentShuffler',
                #     fragment_size=32
                # ),
                # dict(
                #     name='PostProcessSampler',
                #     num=2
                # ),
                # dict(
                #     name='FragmentMirror',
                #     fragment_size=32
                # ),
                # dict(
                #     name='FragmentRotate',
                #     fragment_size=32
                # ),
            ]
        )
        dataset = SingleBranchDataset(video_loader=video_loader, norm=False)
        data = dataset[0]
        print(data)
        # video = torch.from_numpy(np.load("temp.npy"))
        video = data['temporal']
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

    def test_img(self):
        os.chdir('../../')
        img1 = cv2.imread('/data/ly/fragment/0/VQA_ODV/Group1/G1AbandonedKingdom_ERP_7680x3840_fps30_qp27_45406k/23.png')
        img2 = cv2.imread(
            '/data/ly/fragment/0/VQA_ODV/Group1/G1AbandonedKingdom_ERP_7680x3840_fps30_qp27_45406k/25.png')
        cv2.imwrite('test.png',img2-img1)
