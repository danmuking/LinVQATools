"""
从resize成224的视频中随机加载3帧
"""
import os
import random

import cv2
import torch
from data import logger
from data.file_reader.base_reader import BaseReader


class RandomImgReader(BaseReader):
    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix

    def read(self, video_path: str, frame_num=3, is_train: bool = True) -> torch.Tensor:
        """
        读取视频
        Args:
            frame_num: 抽取帧的数量
            video_path: 原始视频路径
            is_train: 是否为训练模式
            return: 成功返回视频，失败返回none
        """
        # 预处理好的视频路径
        video_pre_path = video_path.split('/')
        video_pre_path.insert(3, self.prefix)
        video_pre_path = os.path.join('/', *video_pre_path)[:-4]
        frame = len(os.listdir(video_pre_path))
        video = []

        # 确定要抽取的帧
        random_list = [i for i in range(frame)]
        random_list = random.sample(random_list, frame_num)
        random_list.sort()
        frame_list = random_list

        for i in frame_list:
            img_path = os.path.join(video_pre_path, "{}.png".format(i))
            if not os.path.exists(img_path):
                logger.info("加载{}失败".format(img_path))
                return None
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video.append(torch.tensor(img))
        video = torch.stack(video, dim=0).permute(3, 0, 1, 2)
        logger.info("加载{}成功".format(video_pre_path))
        return video
