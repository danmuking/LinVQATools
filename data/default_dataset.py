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
from data.meta_reader import AbstractReader
from data.split.dataset_split import DatasetSplit
import os.path as osp
import data.sampler as sampler
import decord

random.seed(42)
decord.bridge.set_bridge("torch")


# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)

@DATASETS.register_module()
class DefaultDataset(Dataset):
    def __init__(self, **opt):
        logger = MMLogger.get_instance('mmengine', log_level='INFO')

        # 数据集声明文件根路径
        if 'anno_root' not in opt:
            anno_root = '/home/ly/code/LinVQATools/data/odv_vqa'
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

        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])

    def __getitem__(self, index):
        video_info = self.data[index]
        video_path = video_info["video_path"]
        score = video_info["score"]

        # 直接读取视频
        num = random.randint(0, 79)
        # 预处理好的视频路径
        video_pre_path = video_path.split('/')
        video_pre_path.insert(3, 'fragment')
        video_pre_path.insert(4, '{}'.format(num))
        video_pre_path = os.path.join('/', *video_pre_path)
        if os.path.exists(video_pre_path):
        # if False:
            vreader = VideoReader(video_pre_path)
            frame_dict = {idx: vreader[idx] for idx in range(len(vreader))}
            imgs = [frame_dict[idx] for idx in range(len(vreader))]
            video = torch.stack(imgs, 0).permute(3, 0, 1, 2)
            frame_idxs: List[Any] = []
        else:
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

        video = ((video.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)
        data = {"inputs": video, "num_clips": {},
                # "frame_inds": frame_idxs,
                "gt_label": score,
                "name": osp.basename(video_path)}

        return data

        # return None

    def __len__(self):
        return len(self.data)
