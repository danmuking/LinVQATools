import os

import torch
import random
from typing import Dict, List, Any

import numpy as np
from mmengine import MMLogger
from torch.utils.data import Dataset
from mmengine import DATASETS
from decord import VideoReader
import data.meta_reader as meta_reader
from data.file_reader import ImgReader
from data.meta_reader import AbstractReader
from data.shuffler import BaseShuffler
from data.split.dataset_split import DatasetSplit
import os.path as osp
import data.sampler as sampler
import data.file_reader as reader
import data.shuffler as shuffler
import decord
from data import logger
decord.bridge.set_bridge("torch")


@DATASETS.register_module()
class DefaultDataset(Dataset):
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
        # 视频帧采样器
        self.frame_sampler = getattr(sampler, opt['frame_sampler']['name'])(**opt['frame_sampler'])
        # 空间采样器
        self.spatial_sampler = None
        if 'spatial_sampler' in opt:
            self.spatial_sampler = getattr(sampler, opt['spatial_sampler']['name'])(**opt['spatial_sampler'])
        # 加载预处理文件的加载器
        self.file_reader: ImgReader = getattr(reader, 'ImgReader')(self.prefix)
        self.post_sampler = None
        if 'post_sampler' in opt:
            self.post_sampler = getattr(sampler, opt['post_sampler']['name'])(**opt['post_sampler'])

        self.shuffler:BaseShuffler = getattr(shuffler, opt['shuffler']['name'])(**opt['shuffler'])

        # 读取数据集声明文件
        self.anno_reader: AbstractReader = getattr(meta_reader, opt['anno_reader'])(anno_root)

        # 数据集信息
        self.video_info = self.anno_reader.read()
        # 划分数据集
        self.video_info: Dict = DatasetSplit.split(self.video_info, split_file)

        # 用于获取的训练集/测试集信息
        self.data: List = self.video_info[self.phase]

        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])

    def __getitem__(self, index):
        video_info = self.data[index]
        video_path = video_info["video_path"]
        score = video_info["score"]

        # 含有预处理前缀,加载预处理数据
        if self.phase == 'train':
            video = self.file_reader.read(video_path)
        else:
            video = self.file_reader.read(video_path, is_train=False)
        if self.post_sampler is not None:
            video = self.post_sampler(video)
        # 预处理数据加载失败
        if video is None:
            logger.info("加载未处理的{}".format(video_path))
            vreader = VideoReader(video_path)
            ## Read Original Frames
            ## Process Frames
            frame_idxs = self.frame_sampler(len(vreader))

            ### Each frame is only decoded one time!!!
            all_frame_inds = frame_idxs
            frame_dict = {idx: vreader[idx] for idx in np.unique(all_frame_inds)}
            imgs = [frame_dict[idx] for idx in all_frame_inds]
            video = torch.stack(imgs, 0).permute(3, 0, 1, 2)
            if self.spatial_sampler is not None:
                video = self.spatial_sampler(video)

            # video = self.shuffler.shuffle(video)
        if self.norm:
            video = ((video.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)
        data = {
            "inputs": video, "num_clips": {},
            # "frame_inds": frame_idxs,
            "gt_label": score,
            "name": osp.basename(video_path)
        }

        return data

    def __len__(self):
        return len(self.data)
