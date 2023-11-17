import random

import torch
from einops import rearrange
from mmengine import MMLogger

from data import logger
from .base_shuffler import BaseShuffler


class FragmentShuffler(BaseShuffler):
    """
    从fragment cube上打乱数据
    """

    def __init__(self, fragment_size: int = 32, frame_cube: int = 8, **kargs):
        super().__init__(**kargs)
        self.fragment_size = fragment_size
        self.frame_cube = frame_cube

    def __call__(self, video: torch.Tensor):
        c, t, h, w = video.shape
        p0 = t//self.frame_cube
        p1=h//self.fragment_size
        p2=w//self.fragment_size
        video = rearrange(video, 'c (p0 t) (p1 h) (p2 w) -> c (p0 p1 p2) t h w', p0=p0, p1=p1, p2=p2)
        indices = torch.randperm(video.shape[1])
        video = video[:, indices, :, :, :]
        target_video = rearrange(video, 'c (p0 p1 p2) t h w -> c (p0 t) (p1 h) (p2 w)', p0=p0, p1=p1, p2=p2)
        return target_video
