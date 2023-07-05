"""
生成最不相似的视频碎块
"""
from typing import Dict, List

import torch
from mmengine import MMLogger
from torch.utils.data import Dataset

import data.meta_reader as meta_reader
from data.meta_reader import AbstractReader
from data.split.dataset_split import DatasetSplit
from .similar.video_extractor import VideoExtractor

logger = MMLogger.get_instance('dataset', log_level='INFO')


class GenerateDataset(Dataset):
    """
    数据加载器,方便进行多进程加速
    """
    def __init__(self, **opt):
        # 数据集声明文件根路径
        if 'anno_root' not in opt:
            anno_root = '/home/ly/code/LinVQATools/data/odv_vqa'
            logger.warning("anno_root参数未找到，默认为/home/ly/code/LinVQATools/data/odv_vqa")
        else:
            anno_root = opt['anno_root']

        # 训练集测试集划分文件路径
        split_file = opt.get("split_file", None)
        self.phase = opt.get("phase", 'train')
        # 是否归一化
        self.norm = opt.get('norm', True)
        # 预处理数据前缀
        self.prefix = opt.get('prefix', None)
        self.shuffle = opt.get('shuffle', True)
        # 读取数据集声明文件
        self.anno_reader: AbstractReader = getattr(meta_reader, opt['anno_reader'])(anno_root)

        # 数据集信息
        self.video_info = self.anno_reader.read()
        # 划分数据集
        self.video_info: Dict = DatasetSplit.split(self.video_info, split_file)

        # 用于获取的训练集/测试集信息
        self.data: List = self.video_info[self.phase]

    def __getitem__(self, index):
        video_info = self.data[index]
        video_path = video_info["video_path"]
        VideoExtractor.extract(video_path)
        data = {}

        return data

    def __len__(self):
        return len(self.data)
