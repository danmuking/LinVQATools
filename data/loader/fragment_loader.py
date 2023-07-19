"""
实现fragment视频加载
"""
import numpy as np
import torch
from decord import VideoReader

from data.file_reader.base_reader import BaseReader
from data.loader.base_loader import BaseLoader
from data.shuffler import BaseShuffler
import data.sampler as sampler
import data.file_reader as reader
import data.shuffler as shuffler
import decord
from data import logger

decord.bridge.set_bridge("torch")


class FragmentLoader(BaseLoader):
    def __init__(self, **kwargs):
        super().__init__()
        # 是否归一化
        self.phase = kwargs.get("phase", 'train')
        logger.info("数据加载阶段为:{}".format(self.phase))
        self.norm = kwargs.get('norm', True)
        logger.info("fragment数据加载器是否使用归一化:{}".format(self.norm))
        # 预处理数据前缀
        self.prefix = kwargs.get('prefix', None)
        logger.info("fragment数据加载器预处理数据前缀:{}".format(self.prefix))
        # 视频帧采样器
        self.frame_sampler = getattr(sampler, kwargs['frame_sampler']['name'])(**kwargs['frame_sampler'])
        # 空间采样器
        self.spatial_sampler = None
        if 'spatial_sampler' in kwargs:
            self.spatial_sampler = getattr(sampler, kwargs['spatial_sampler']['name'])(**kwargs['spatial_sampler'])
        # 加载预处理数据的加载器
        self.file_reader: BaseReader = getattr(reader, 'ImgReader')(self.prefix)
        # 数据后处理策略
        self.post_sampler = None
        if 'post_sampler' in kwargs:
            self.post_sampler = getattr(sampler, kwargs['post_sampler']['name'])(**kwargs['post_sampler'])
        # 数据增强策略 
        self.shuffler: BaseShuffler = getattr(shuffler, kwargs['shuffler']['name'])(**kwargs['shuffler'])

        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])

    def read(self, path: str) -> torch.Tensor:
        video_path = path
        if self.phase == 'train':
            video = self.file_reader.read(video_path)
        else:
            video = self.file_reader.read(video_path, False)
        logger.debug("加载视频数据维度为:{}".format(video.size()))

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

        # 后处理
        if self.post_sampler is not None:
            video = self.post_sampler(video)
            logger.debug("后处理后视频数据维度为:{}".format(video.size()))
        # 打乱
        video = self.shuffler.shuffle(video)
        # 归一化
        if self.norm:
            video = ((video.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)

        return video
