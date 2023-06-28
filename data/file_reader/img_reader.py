"""
从png图片中加载视频
"""
import os
import random

import cv2
import torch


class ImgReader:
    def __init__(self, prefix):
        self.prefix = prefix

    def read(self, video_path: str, is_train: bool = True) -> torch.Tensor:
        """
        读取视频
        :param video_path: 原始视频路径
        :param is_train: 是否为训练模式
        :return: 成功返回视频，失败返回none
        """
        # 直接读取视频
        if is_train:
            num = random.randint(0, 150)
        else:
            num = 0
        # 预处理好的视频路径
        video_pre_path = video_path.split('/')
        video_pre_path.insert(3, self.prefix)
        video_pre_path.insert(4, '{}'.format(num))
        video_pre_path = os.path.join('/', *video_pre_path)[:-4]
        video = []
        for i in range(32):
            img_path = os.path.join(video_pre_path, "{}.png".format(i))
            if not os.path.exists(img_path):
                return None
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video.append(torch.tensor(img))
        video = torch.stack(video, dim=0).permute(3, 0, 1, 2)
        return video
