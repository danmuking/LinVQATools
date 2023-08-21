"""
将视频划分为k个视频块,从每个视频块中随机抽取一帧,共抽取k帧
Examples:
    video = [1,2,3,4,5,6]
    k = 2
    video_cube = [[1 2 3] [4 5 6]]
    result perhaps equal [1 5]
"""
import os
from typing import List, Dict

import torch
from einops import rearrange

from data.file_reader.image_reader import ImageReader
from data.loader.base_loader import BaseLoader
from data import sampler as sampler
from data import shuffler


class VideoSamplerLoader(BaseLoader):
    def __init__(
            self,
            frame_sampler: str = 'CubeExtractSample',
            k: int = 16,
            use_preprocess=True,
            prefix='fragment',
            argument: List[Dict] = [],
            *args,
            **kwargs,
    ):
        self.frame_sampler = getattr(sampler, frame_sampler)(k)
        self.use_preprocess = use_preprocess
        self.prefix = prefix
        self.argument = [getattr(shuffler, item['name'])(**item) for item in argument]

    def __call__(self, video_path: str, frame_num: int, *args, **kwargs):
        # 帧采样
        frame_indexs = self.frame_sampler(frame_num)
        video = []
        if self.use_preprocess:
            # 使用预处理文件视频加载
            video_path = video_path.split('/')
            video_path.insert(3, 'fragment')
            video_path[0] = "/data"
            video_path[1] = ""
            video_path = os.path.join(*video_path)[:-4]
            for frame_index in frame_indexs:
                # 计算图片地址
                img_path = os.path.join(video_path, '{}.png'.format(frame_index))
                img = ImageReader.read(img_path)
                video.append(img)
        else:
            # TODO: 不使用预处理
            pass
        video = torch.stack(video, dim=0)
        video = rearrange(video, 't h w c -> c t h w')
        # 视频后处理
        for item in self.argument:
            video = item(video)
        return video
