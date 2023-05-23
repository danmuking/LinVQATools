import torch
import random
from typing import Dict, List

import numpy as np
from mmengine import MMLogger
from torch.utils.data import Dataset
from mmengine import DATASETS
from decord import VideoReader
import data.meta_reader as meta_reader
from data.meta_reader import AbstractReader
from data.split.dataset_split import DatasetSplit
import os.path as osp
import data.sampler as sampler
import decord

random.seed(42)
decord.bridge.set_bridge("torch")


@DATASETS.register_module()
class DefaultDataset(Dataset):
    def __init__(self, **opt):
        logger = MMLogger.get_instance('mmengine', log_level='INFO')

        # 数据集声明文件根路径
        if 'anno_root' not in opt:
            anno_root = './data/odv_vqa'
            logger.warning("anno_root参数未找到，默认为./data/odv_vqa")
        else:
            anno_root = opt['anno_root']

        # 训练集测试集划分文件路径
        split_file = opt.get("split_file", None)
        self.phase = opt.get("phase", 'train')
        # 视频帧采样器
        self.frame_sampler = getattr(sampler, opt['frame_sampler']['name'])(**opt['frame_sampler'])
        # 空间采样器
        self.spatial_sampler = None
        if 'spatial_sampler' in opt:
            self.spatial_sampler = getattr(sampler, opt['spatial_sampler']['name'])(**opt['spatial_sampler'])

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
        score = video_info["score"]

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
        data = {"inputs": video, "num_clips": {}, "frame_inds": frame_idxs, "gt_label": score,
                "name": osp.basename(video_path)}
        # for k, v in data.items():
        #     data[k] = ((v.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)

        # for stype, sopt in self.sample_types.items():
        #     data["num_clips"][stype] = sopt["num_clips"]
        # print(data['fragments'].shape)
        # data = dict(
        #     inputs=data,
        # )
        return data

    def __len__(self):
        return len(self.data)
