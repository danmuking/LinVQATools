"""
实现fragment视频加载
"""
from typing import List, Dict

import numpy as np
import torch
from decord import VideoReader

from data.file_reader.base_reader import BaseReader
from data.loader.base_loader import BaseLoader
from data.shuffler import BaseShuffler, SpatialShuffler
import data.sampler as sampler
import data.file_reader as reader
import data.shuffler as shuffler
import decord
from data import logger

decord.bridge.set_bridge("torch")


class FragmentLoader(BaseLoader):
    def __init__(
            self,
            prefix='temp/fragment',
            frame_sampler=None,
            spatial_sampler=None,
            argument: List[Dict] = [],
            phase='train',
            use_preprocess=True,
            **kwargs):
        super().__init__()
        # 是否归一化
        self.phase = phase
        # 是否使用预训练数据
        self.use_preprocess = use_preprocess
        # 预处理数据前缀
        self.prefix = prefix
        # 加载预处理数据的加载器
        self.file_reader: BaseReader = getattr(reader, 'ImgReader')(self.prefix)
        # 数据增强
        self.argument = [getattr(shuffler, item['name'])(**item) for item in argument]
        # 视频帧采样器
        self.frame_sampler = frame_sampler
        if self.frame_sampler is not None:
            self.frame_sampler = getattr(sampler, frame_sampler['name'])(**frame_sampler)
        # 空间采样器
        self.spatial_sampler = spatial_sampler
        if self.spatial_sampler is not None:
            self.spatial_sampler = getattr(sampler, spatial_sampler['name'])(**spatial_sampler)

    def __call__(self, video_path: str,*args, **kwargs) -> torch.Tensor:
        if self.use_preprocess:
            if self.phase == 'train':
                video = self.file_reader.read(video_path)
            else:
                video = self.file_reader.read(video_path, is_train=False)
            logger.debug("加载视频数据维度为:{}".format(video.size()))
        else:
            # 预处理数据加载失败
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
        # 视频后处理
        argument = SpatialShuffler()
        video,pos_embed = argument(video)
        for item in self.argument:
            video = item(video)
        return video,pos_embed
