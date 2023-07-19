"""
从png图片中加载视频
"""
import os
import random

import cv2
import torch
from data import logger
from data.file_reader.base_reader import BaseReader


class ImgReader(BaseReader):
    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix

    def read(self, video_path: str, cube_num:int=4, is_train: bool = True) -> None:
        """
        读取视频
        Args:
            cube_num: 抽取时间块的数量
            video_path: 原始视频路径
            is_train: 是否为训练模式
            return: 成功返回视频，失败返回none
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

        # 确定要抽取的帧
        random_list = [i for i in range(4)]
        random_list = random.sample(random_list, cube_num)
        random_list.sort()
        frame_list = []
        # 每个cube包含8帧
        cube_frame = 8
        for i in random_list:
            for j in range(cube_frame):
                frame_list.append(cube_frame*i+j)

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
