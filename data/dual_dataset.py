"""
双分支网络的数据加载
"""
import os
from typing import Dict, List

import torch
from mmengine import DATASETS
from torch.utils.data import Dataset
from data import loader as loader
from data import logger
from data.meta_reader import AbstractReader
from data.split.dataset_split import DatasetSplit
import data.meta_reader as meta_reader


@DATASETS.register_module()
class DualDataset(Dataset):
    """
    双分支网络数据集加载器
    """

    def __init__(self, loader1: Dict,
                 loader2: Dict,
                 anno_root: str = './data/odv_vqa',
                 anno_reader: str = 'ODVVQAReader',
                 split_file: str = './data/odv_vqa/tr_te_VQA_ODV.txt',
                 phase: str = 'train',
                 norm: bool = True
                 ):
        """
        初始化
        Args:
            loader1: 加载器1
            loader2: 加载器2
        """
        # 数据集声明文件夹路径
        self.anno_root = anno_root
        # 数据集声明文件加载器
        self.anno_reader: AbstractReader = getattr(meta_reader, anno_reader)(anno_root)

        # 训练集/测试集划分文件
        self.split_file = split_file
        # 训练集/测试集
        self.phase = phase
        # 是否归一化
        self.norm = norm

        # 数据集信息
        self.video_info = self.anno_reader.read()
        # 划分数据集
        self.video_info: Dict = DatasetSplit.split(self.video_info, split_file)

        # 用于获取的训练集/测试集信息
        self.data: List[Dict] = self.video_info[self.phase]

        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])

        self.loader1 = getattr(loader, loader1['name'])(**loader1)
        self.loader2 = getattr(loader, loader2['name'])(**loader2)

    def __getitem__(self, index):
        video_info = self.data[index]
        video_path = video_info["video_path"]
        score = video_info["score"]
        video = [self.loader1.read(video_path), self.loader2.read(video_path)]

        data = {
            "inputs": video, "num_clips": {},
            # "frame_inds": frame_idxs,
            "gt_label": score,
            "name": os.path.basename(video_path)
        }
        return data

    def __len__(self):
        return len(self.data)
