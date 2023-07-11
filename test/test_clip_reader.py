from unittest import TestCase

import cv2
import numpy as np

from data.file_reader.clip_reader import ClipReader


class TestClipReader(TestCase):
    def test_read(self):
        reader = ClipReader()
        reader.read('/data/ly/VQA_ODV/Group10/Reference/G10PandaBaseChengdu_7680x3840_fps29.97.mp4')
    def test_save(self):
        reader = ClipReader()
        video = reader.read('/data/ly/VQA_ODV/Group9/G9FootballMatch_ERP_4096x2048_fps30_qp42_288k.mp4')
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
