import numpy as np
import torch
from typing import Dict, List, Any

from decord import VideoReader
from einops import rearrange
from torch.utils.data import Dataset
from mmengine import DATASETS
import os.path as osp
import decord
from torchvision.transforms import transforms

from data import logger
import data.meta_reader as meta_reader
from data.meta_reader import AbstractReader
from data.split.dataset_split import DatasetSplit
from data import loader

decord.bridge.set_bridge("torch")


def frame_compute(data):
    data = [x['frame_num'] for x in data]
    data.append(1)
    frame_index = np.cumsum(data)
    frame_index = np.insert(frame_index, 0, 0)
    return frame_index


@DATASETS.register_module()
class ImageDataset(Dataset):
    """
    视频帧数据加载器
    """

    def __init__(
            self,
            anno_root: str = './data/odv_vqa',
            anno_reader: str = 'ODVVQAReader',
            split_file: str = './data/odv_vqa/tr_te_VQA_ODV.txt',
            phase: str = 'train',
            norm: bool = True

    ):
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

        self.frame_index = frame_compute(self.data)

        self.frame_interval = 10

        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])

        # 视频中相机是否移动
        self.camera_motion = [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1,
                              1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0]

        self.data_argument = None
        means, stds = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if phase == 'train':
            self.data_argument = transforms.Compose([
                transforms.Resize(512),
                transforms.RandomCrop(320),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Normalize(means, stds),
            ])
        else:
            self.data_argument = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(320),
                # transforms.RandomHorizontalFlip(0.5),
                transforms.Normalize(means, stds),
            ])

    def __getitem__(self, index):
        frame_index = index * self.frame_interval
        # 获取视频索引
        video_index = None
        for i,video_range in enumerate(self.frame_index[1:]):
            if frame_index < video_range:
                video_index = i
                break
        video_info = self.data[video_index]
        video_path: str = video_info["video_path"]
        score = video_info["score"]
        frame_num = video_info['frame_num']
        img = VideoReader(video_path)[frame_index-self.frame_index[i]]
        img = rearrange(img, 'h w c -> c h w')
        img = img/255
        img = self.data_argument(img)
        camera_motion = self.camera_motion[video_info['scene_id']]
        data = {
            "inputs": img, "num_clips": {},
            # "frame_inds": frame_idxs,
            "gt_label": score,
            'camera_motion': camera_motion,
            "name": osp.basename(video_path)
        }

        return data

    def __len__(self):
        return int(self.frame_index[-1]/self.frame_interval)
