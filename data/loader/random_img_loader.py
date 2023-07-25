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


class RandomImgLoader(BaseLoader):
    def __init__(self, **kwargs):
        super().__init__()
        # 是否归一化
        self.norm = kwargs.get('norm', True)
        logger.info("random img数据加载器是否使用归一化:{}".format(self.norm))
        # 预处理数据前缀
        self.prefix = kwargs.get('prefix', None)
        logger.info("random img数据加载器预处理数据前缀:{}".format(self.prefix))
        # 加载预处理数据的加载器
        self.file_reader: BaseReader = getattr(reader, 'RandomImgReader')(self.prefix)
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
        video = self.file_reader.read(video_path)
        logger.debug("加载视频数据维度为:{}".format(video.size()))

        # # 预处理数据加载失败
        # pass

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
