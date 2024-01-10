import random

import torch
from typing import Dict, List, Any

from decord import VideoReader
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

decord.bridge.set_bridge("native")


@DATASETS.register_module()
class SingleBranchDataset(Dataset):
    """
    单分支网络数据加载器
    """

    def __init__(
            self,
            video_loader: Dict,
            anno_root: str = './data/odv_vqa',
            anno_reader: str = 'ODVVQAReader',
            split_file: str = './data/odv_vqa/tr_te_VQA_ODV.txt',
            phase: str = 'train',
            norm: bool = True,
            clip=1,
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
        self.clip = clip

        # 数据集信息
        self.video_info = self.anno_reader.read()
        # 划分数据集
        self.video_info: Dict = DatasetSplit.split(self.video_info, split_file)

        # 用于获取的训练集/测试集信息
        self.data: List[Dict] = self.video_info[self.phase]

        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
        self.std = torch.FloatTensor([0.229, 0.224, 0.225])

        # 视频加载器
        self.video_loader = getattr(loader, video_loader['name'])(**video_loader)

        self.img_transform = transforms.Compose([
                                transforms.ToPILImage(mode='RGB'),
                                transforms.Resize(448),
                                # transforms.RandomHorizontalFlip(),
                                transforms.CenterCrop(224),
                                # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean, self.std)
                            ])

    def __getitem__(self, index):
        video_info = self.data[index]
        video_path: Dict = video_info["video_path"]
        vr = VideoReader(video_path)
        score = video_info["score"]
        frame_num = video_info['frame_num']

        videos = []
        imgs = []
        for i in range(self.clip):
            video = self.video_loader(video_path=video_path, frame_num=frame_num)
            videos.append(video)

            frame_index = random.randint(0, len(vr)-1)
            # 交换维度
            img = vr[frame_index].asnumpy()
            # print(img.shape)
            img = self.img_transform(img)
            imgs.append(img)

        video = torch.stack(videos, dim=0)
        img=torch.stack(imgs,dim=0)
        raw_video = video
        if self.norm:
            video = video / 255.0
            video = ((video.permute(0, 2, 3, 4, 1) - self.mean) / self.std).permute(0,4, 1, 2, 3)
        data = {
            "inputs": {'video':video,'img':img},
            "raw_video": raw_video,
            "num_clips": self.clip,
            # "frame_inds": frame_idxs,
            "gt_label": score,
            "name": osp.basename(video_path)
        }

        return data

    def __len__(self):
        return len(self.data)
