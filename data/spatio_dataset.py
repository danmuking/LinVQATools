import os
from functools import lru_cache

import cv2
import torch
import random
from typing import Dict, List, Any

import numpy as np
import torchvision
from einops import rearrange
from mmengine import MMLogger
from sympy import prime
from torch.utils.data import Dataset
from mmengine import DATASETS
from decord import VideoReader
from torchvision.transforms import ToPILImage

import data.meta_reader as meta_reader
from data.file_reader import ImgReader
from data.meta_reader import AbstractReader
from data.sampler.time_fragment_sampler import UnifiedFrameSampler
from data.split.dataset_split import DatasetSplit
import os.path as osp
import data.sampler as sampler
import data.file_reader as reader
import decord

from utils.data_preprocess import frame

decord.bridge.set_bridge("torch")

# 日志器
logger = MMLogger.get_instance('dataset', log_level='INFO')


# @DATASETS.register_module()
class SpatioDataset(Dataset):
    """
    用于进行resize
    """

    def __init__(self, **opt):

        # 数据集声明文件根路径
        if 'anno_root' not in opt:
            anno_root = '/home/ly/data/code/LinVQATools/data/odv_vqa'
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
        views = process(video_path)
        # views = 0
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


def process(video_path):
    vreader = VideoReader(video_path)
        # 获取视频的总帧数
    num_frames = len(vreader)

    # 获取视频的分辨率（宽度和高度）
    width, height = vreader[0].shape[1], vreader[0].shape[0]
    patch_size = 32
    frame_nums = random.sample(range(0, num_frames), 50)
    raw_frame_list = [vreader[i] for i in frame_nums]
    # raw_frame_list = [torch.zeros(3840, 7680, 3) for i in frame_nums]
    img_list = []
    for i in range(32):
        h_start = random.randint(0, height-patch_size)
        w_start = random.randint(0, width-patch_size)
        frames = random.sample(raw_frame_list, 25)
        patch_list = []
        for each_frame in frames:
            # print(each_frame.shape)
            each_frame = each_frame[h_start:h_start+patch_size,w_start:w_start+patch_size,...]
            patch_list.append(each_frame)
        patchs = torch.stack(patch_list,dim=0)
        # print(patchs.shape)
        # 展开frame
        img = rearrange(patchs, '(ih iw) h w c -> (ih h) (iw w) c',ih=5,iw=5).permute(2, 0, 1)
        # print("img:"+str(img.shape))
        img_list.append(img)

    return img_list


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
