"""
双分支网络的数据加载
"""
import os
import random
from typing import Dict, List

import cv2
import torch
from mmengine import DATASETS
from torch.utils.data import Dataset
from data import loader as loader
from data import logger
from data.meta_reader import AbstractReader
from data.split.dataset_split import DatasetSplit
import data.meta_reader as meta_reader


@DATASETS.register_module()
class GeneratorDataset(Dataset):
    """
    生成网络数据加载器
    """

    def __init__(
            self,
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

        # self.loader1 = getattr(loader, loader1['name'])(**loader1)
        # self.loader2 = getattr(loader, loader2['name'])(**loader2)

    def __getitem__(self, index):
        video_info = self.data[index]
        video_path = video_info["video_path"]
        score = video_info["score"]

        prefix = 'imp_ref'
        # 直接读取视频
        if self.phase:
            num = random.randint(0, 39)
        else:
            num = 0
        num = 0
        ########## impair video ###########
        # 预处理好的视频路径
        video_pre_path = video_path.split('/')
        video_pre_path.insert(3, prefix)
        video_pre_path.insert(4, '{}'.format(num))
        video_pre_path = os.path.join('/', *video_pre_path)[:-4]
        logger.debug("尝试加载{}".format(video_pre_path))
        img_list = os.listdir(video_pre_path)
        frame_index = []
        for x in img_list:
            if x != 'ref':
                frame_index.append(int(x[:-4]))
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

        ########## ref video ###########
        # 预处理好的视频路径
        video_pre_path = video_path.split('/')
        video_pre_path.insert(3, prefix)
        video_pre_path.insert(4, '{}'.format(num))
        video_pre_path = os.path.join('/', *video_pre_path)[:-4]
        video_pre_path = os.path.join(video_pre_path, 'ref')
        logger.debug("尝试加载{}".format(video_pre_path))
        img_list = os.listdir(video_pre_path)
        frame_index = []
        for x in img_list:
            if x != 'ref':
                frame_index.append(int(x[:-4]))
        frame_index.sort()
        ref_video = []
        for i in frame_index:
            img_path = os.path.join(video_pre_path, "{}.png".format(i))
            if not os.path.exists(img_path):
                logger.error("{}加载失败".format(img_path))
                return None
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ref_video.append(torch.tensor(img))
        ref_video = torch.stack(ref_video, dim=0).permute(3, 0, 1, 2)
        ref_video = ref_video - video
        ref_video= ref_video[:, ::2, ...]

        data = {
            "inputs": video, "num_clips": {},
            'gt_video': ref_video,
            # "frame_inds": frame_idxs,
            "gt_label": score,
            "name": os.path.basename(video_path)
        }
        return data

    def __len__(self):
        return len(self.data)
