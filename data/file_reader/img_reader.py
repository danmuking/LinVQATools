"""
从png图片中加载视频
"""
import os
import random

import cv2
import torch
from data import logger

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
            # num = 0
        else:
            num = 0
        # 预处理好的视频路径
        video_pre_path = video_path.split('/')
        video_pre_path.insert(3, self.prefix)
        video_pre_path.insert(4, '{}'.format(num))
        video_pre_path = os.path.join('/', *video_pre_path)[:-4]
        logger.debug("尝试加载{}".format(video_pre_path))
        img_list = os.listdir(video_pre_path)
        frame_index = [int(x[:-4]) for x in img_list]
        frame_index.sort()
        video = []
        for i in frame_index:
            img_path = os.path.join(video_pre_path, "{}.png".format(i))
            if not os.path.exists(img_path):
                logger.error("{}加载失败".format(img_path))
                return None
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video.append(torch.tensor(img))
        video = torch.stack(video, dim=0).permute(3, 0, 1, 2)
        return video
