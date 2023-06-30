import os
from functools import lru_cache

import cv2
import torch
import random
from typing import Dict, List, Any

import numpy as np
import torchvision
from mmengine import MMLogger
from torch.utils.data import Dataset
from mmengine import DATASETS
from decord import VideoReader
import data.meta_reader as meta_reader
from data.file_reader import ImgReader
from data.meta_reader import AbstractReader
from data.sampler.time_fragment_sampler import UnifiedFrameSampler
from data.split.dataset_split import DatasetSplit
import os.path as osp
import data.sampler as sampler
import data.file_reader as reader
import decord

decord.bridge.set_bridge("torch")

# 日志器
logger = MMLogger.get_instance('dataset', log_level='INFO')


# @DATASETS.register_module()
class ResizeDataset(Dataset):
    """
    用于进行resize
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
        # 视频帧采样器
        self.frame_sampler = getattr(sampler, opt['frame_sampler']['name'])(**opt['frame_sampler'])
        # 空间采样器
        self.spatial_sampler = None
        if 'spatial_sampler' in opt:
            self.spatial_sampler = getattr(sampler, opt['spatial_sampler']['name'])(**opt['spatial_sampler'])
        # 加载预处理文件的加载器
        self.file_reader: ImgReader = getattr(reader, 'ImgReader')(self.prefix)
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
        # ----------------------------实现aesthetic----------------------------------------
        # video = self.video_reader()
        temporal_samplers = dict()
        temporal_samplers['aesthetic'] = UnifiedFrameSampler(
            1, 32, 2, 1
        )
        views, _ = spatial_temporal_view_decomposition(
            video_path, {'aesthetic': {'size_h': 224, 'size_w': 224, 'clip_len': 32, 'frame_interval': 2, 't_frag': 32,
                                       'num_clips': 1}}, temporal_samplers
        )
        # -------------------------------------------------------------------------------------------
        data = {
            "inputs": views, "num_clips": {},
            # "frame_inds": frame_idxs,
            "gt_label": score,
            "name": osp.basename(video_path)
        }

        return data

        # return None

    def __len__(self):
        return len(self.data)


def spatial_temporal_view_decomposition(
        video_path, sample_types, samplers, is_train=False, augment=False,
):
    video = None
    vreader = VideoReader(video_path)
    ### Avoid duplicated video decoding!!! Important!!!!
    all_frame_inds = []
    frame_inds = {}
    for stype in samplers:
        frame_inds[stype] = samplers[stype](len(vreader), is_train)
        all_frame_inds.append(frame_inds[stype])
    all_frame_inds = [i for i in range(len(vreader))]
    ### Each frame is only decoded one time!!!
    all_frame_inds = np.array(all_frame_inds)

    video_pre_path = video_path.split('/')
    video_pre_path.insert(3, 'resize')
    video_pre_path = os.path.join('/', *video_pre_path)[:-4]
    imgs = []
    video = {}
    # frame_dict = {idx: vreader[idx] for idx in np.unique(all_frame_inds)}
    temp_list = []
    for stype in samplers:
        for i in all_frame_inds:
            imgs = vreader[i]
            imgs = imgs.unsqueeze(0)
            video[stype] = imgs.permute(3, 0, 1, 2)
            # print(video[stype].shape)
            temp = get_single_view(video[stype], stype, **{
                'aesthetic': {'size_h': 224, 'size_w': 224, 'clip_len': 32, 'frame_interval': 2, 't_frag': 32,
                              'num_clips': 1}})
            temp_list.append(temp)

    sampled_video=temp_list

    return sampled_video, frame_inds


def get_single_view(
        video, sample_type="aesthetic", **kwargs,
):
    if sample_type.startswith("aesthetic"):
        video = get_resized_video(video, **kwargs)
    elif sample_type == "original":
        return video

    return video


def get_resized_video(
        video, size_h=224, size_w=224, random_crop=False, arp=False, **kwargs,
):
    video = video.permute(1, 0, 2, 3)
    resize_opt = get_resize_function(
        size_h, size_w, video.shape[-2] / video.shape[-1] if arp else 1, random_crop
    )
    video = resize_opt(video).permute(1, 0, 2, 3)
    return video


@lru_cache
def get_resize_function(size_h, size_w, target_ratio=1, random_crop=False):
    if random_crop:
        return torchvision.transforms.RandomResizedCrop(
            (size_h, size_w), scale=(0.40, 1.0)
        )
    if target_ratio > 1:
        size_h = int(target_ratio * size_w)
        assert size_h > size_w
    elif target_ratio < 1:
        size_w = int(size_h / target_ratio)
        assert size_w > size_h
    return torchvision.transforms.Resize((size_h, size_w))
