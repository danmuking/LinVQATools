"""
从torch矩阵中读取视频
"""
import os
import random

import torch


class TorchReader:
    def __init__(self, prefix):
        self.prefix = prefix

    def read(self, video_path: str, is_train: bool = True) -> torch.Tensor:
        # 直接读取视频
        if is_train:
            num = random.randint(0, 39)
        else:
            num = 0
        # 预处理好的视频路径
        video_pre_path = video_path.split('/')
        video_pre_path.insert(3, self.prefix)
        video_pre_path.insert(4, '{}'.format(num))
        video_pre_path = os.path.join('/', *video_pre_path)
        video = torch.load(video_pre_path)
        return video
