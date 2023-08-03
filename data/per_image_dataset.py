"""
数据集加载器，实现将每一帧作为单独的输入进行回归
"""
import os
import random
from typing import Dict, List

import cv2
import torch
import decord
import numpy as np
from decord import VideoReader
from einops import rearrange
from mmengine import DATASETS
from torch.utils.data import Dataset

from data import logger, meta_reader
from data.meta_reader import AbstractReader
from data.split.dataset_split import DatasetSplit

decord.bridge.set_bridge("torch")


@DATASETS.register_module()
class PerImageDataset(Dataset):
    """
    单帧数据集加载器
    """

    def __init__(self, **opt: Dict):
        """
        初始化
        Args:
        """
        # 数据集声明文件根路径
        if 'anno_root' not in opt:
            anno_root = './data/odv_vqa'
            logger.warning("anno_root参数未找到，默认为/home/ly/code/LinVQATools/data/odv_vqa")
        else:
            anno_root = opt['anno_root']

        # 训练集测试集划分文件路径
        split_file = opt.get("split_file", None)
        # 读取数据集声明文件
        self.anno_reader: AbstractReader = getattr(meta_reader, opt['anno_reader'])(anno_root)
        self.phase = opt.get("phase", 'train')
        # 数据集信息
        self.video_info = self.anno_reader.read()
        # 划分数据集
        self.video_info: Dict = DatasetSplit.split(self.video_info, split_file)
        # 用于获取的训练集/测试集信息
        self.data: List = self.video_info[self.phase]

        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])

        self.prefix = opt.get("prefix", 'train')

    def __getitem__(self, index):
        video_index = index // 32
        video_info = self.data[video_index]
        index = index % 32
        video_path = video_info["video_path"]
        score = video_info["score"]

        # 直接读取视频
        if self.phase == 'train':
            num = random.randint(0, 150)
        else:
            num = 0
        # 预处理好的视频路径
        video_pre_path = video_path.split('/')
        video_pre_path.insert(3, self.prefix)
        video_pre_path.insert(4, '{}'.format(num))
        video_pre_path = os.path.join('/', *video_pre_path)[:-4]
        video = []
        img_path = os.path.join(video_pre_path, "{}.png".format(index))
        if not os.path.exists(img_path):
            logger.error("{}加载失败".format(img_path))
            return None
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img)
        img = rearrange(img, 'h w c -> c h w')

        img = ((img.permute(1, 2, 0) - self.mean) / self.std).permute(2, 0, 1)

        data = {
            "inputs": img,
            "gt_label": score,
            "name": os.path.basename(video_path)
        }
        return data

    def __len__(self):
        return len(self.data) * 32
