"""
数据集加载器，实现将每一帧作为单独的输入进行回归
"""
import os
import random
from typing import Dict, List

import torch
import decord
import numpy as np
from decord import VideoReader
from einops import rearrange
from mmengine import DATASETS
from torch.utils.data import Dataset

from data import logger, meta_reader
from data.meta_reader import AbstractReader
from data.split.dataset_split import DatasetSplit

decord.bridge.set_bridge("torch")


@DATASETS.register_module()
class PerImageDataset(Dataset):
    """
    单帧数据集加载器
    """

    def __init__(self, **opt: Dict):
        """
        初始化
        Args:
        """
        # 数据集声明文件根路径
        if 'anno_root' not in opt:
            anno_root = './data/odv_vqa'
            logger.warning("anno_root参数未找到，默认为/home/ly/code/LinVQATools/data/odv_vqa")
        else:
            anno_root = opt['anno_root']

        # 训练集测试集划分文件路径
        split_file = opt.get("split_file", None)
        # 读取数据集声明文件
        self.anno_reader: AbstractReader = getattr(meta_reader, opt['anno_reader'])(anno_root)
        self.phase = opt.get("phase", 'train')
        # 数据集信息
        self.video_info = self.anno_reader.read()
        # 划分数据集
        self.video_info: Dict = DatasetSplit.split(self.video_info, split_file)
        # 用于获取的训练集/测试集信息
        self.data: List = self.video_info[self.phase]

        frame_index = np.array([item['frame'] for item in self.data])
        self.frame_index = np.cumsum(frame_index)
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])

    def __getitem__(self, index):
        video_index = 0
        for _, item in enumerate(self.frame_index):
            video_index = _
            if index - item <= 0:
                break
        video_info = self.data[video_index]
        index = index - self.frame_index[video_index] + video_info['frame']
        video_path = video_info["video_path"]
        score = video_info["score"]
        vreader = VideoReader(video_path)
        img = vreader[index]
        img = rearrange(img, 'h w c -> c h w')
        # print(img.shape)

        #############################################################################################
        fragments_h = 7
        fragments_w = 7
        fsize_h = 32
        fsize_w = 32
        # 采样图片的高
        size_h = fragments_h * fsize_h
        # 采样图片的长
        size_w = fragments_w * fsize_w

        res_h, res_w = img.shape[-2:]
        size = size_h, size_w

        ## make sure that sampling will not run out of the picture
        hgrids = torch.LongTensor(
            [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
        )
        wgrids = torch.LongTensor(
            [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
        )
        hlength, wlength = res_h // fragments_h, res_w // fragments_w
        if hlength > fsize_h:
            rnd_h = torch.randint(
                hlength - fsize_h, (len(hgrids), len(wgrids))
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids)).int())
        if wlength > fsize_w:
            rnd_w = torch.randint(
                wlength - fsize_w, (len(hgrids), len(wgrids))
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids)).int())

        target_img = torch.zeros((3,224,224))

        for i, hs in enumerate(hgrids):
            for j, ws in enumerate(wgrids):
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                h_so, h_eo = hs + rnd_h[i][j], hs + rnd_h[i][j] + fsize_h
                w_so, w_eo = ws + rnd_w[i][j], ws + rnd_w[i][j] + fsize_w
                # print(h_so, w_so)
                target_img[:, h_s:h_e, w_s:w_e] = img[:, h_so:h_eo, w_so:w_eo]

        target_img = ((target_img.permute(1, 2, 0) - self.mean) / self.std).permute(2, 0, 1)

        data = {
            "inputs": target_img,
            "gt_label": score,
            "name": os.path.basename(video_path)
        }
        return data

    def __len__(self):
        return self.frame_index[-1]
