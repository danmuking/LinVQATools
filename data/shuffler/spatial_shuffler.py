import random

import torch
from einops import rearrange
from mmengine import MMLogger

from data import logger
from models.backbones.video_mae_v2 import get_sinusoid_encoding_table
from .base_shuffler import BaseShuffler


class SpatialShuffler(BaseShuffler):
    """
    实现数据空间维度上的打乱
    """

    def __init__(self, fragment_size: int = 32, **kargs):
        super().__init__(**kargs)
        self.fragment_size = fragment_size

    def __call__(self, video: torch.Tensor):
        c, t, h, w = video.shape
        logger.debug("从空间维度打乱视频")
        matrix = []
        assert w % self.fragment_size == 0, '视频宽度无法被fragment_size整除,W:{} fragment_size:{}'.format(w,
                                                                                                           self.fragment_size)
        assert h % self.fragment_size == 0, '视频高度无法被fragment_size整除,H:{} fragment_size:{}'.format(h,
                                                                                                           self.fragment_size)
        num_w = w // self.fragment_size
        num_h = h // self.fragment_size
        for i in range(num_h):
            for j in range(num_w):
                matrix.append((i, j))
        random.shuffle(matrix)
        count = 0
        target_video = torch.zeros_like(video)
        for i in range(num_w):
            for j in range(num_h):
                h_s, h_e = i * self.fragment_size, (i + 1) * self.fragment_size
                w_s, w_e = j * self.fragment_size, (j + 1) * self.fragment_size
                h_so, h_eo = matrix[count][0] * self.fragment_size, (matrix[count][0] + 1) * self.fragment_size
                w_so, w_eo = matrix[count][1] * self.fragment_size, (matrix[count][1] + 1) * self.fragment_size
                target_video[:, :, h_s:h_e, w_s:w_e] = video[
                                                       :, :, h_so:h_eo, w_so:w_eo
                                                       ]
                count = count + 1

        pos_embed = get_sinusoid_encoding_table(1568, 384)
        pos_embed = rearrange(pos_embed, 'b (t h w) c -> (b c) t h w',t=8, h=14, w=14)
        target_pos_embed = torch.zeros_like(pos_embed)
        count = 0
        for i in range(num_w):
            for j in range(num_h):
                h_s, h_e = i * 2, (i + 1) * 2
                w_s, w_e = j * 2, (j + 1) * 2
                h_so, h_eo = matrix[count][0] * 2, (matrix[count][0] + 1) * 2
                w_so, w_eo = matrix[count][1] * 2, (matrix[count][1] + 1) * 2
                target_pos_embed[:, :, h_s:h_e, w_s:w_e] = pos_embed[
                                                       :, :, h_so:h_eo, w_so:w_eo
                                                       ]
                count = count + 1
        target_pos_embed = rearrange(target_pos_embed, 'c t h w -> (t h w) c',t=8, h=14, w=14)
        return target_video,target_pos_embed


class MixShuffler(BaseShuffler):
    def __init__(self, fragment_size: int = 32, **kargs):
        super().__init__(**kargs)
        self.fragment_size = fragment_size

    def __call__(self, video: torch.Tensor):
        c, t, h, w = video.shape
        fragment_num = h // self.fragment_size
        fragment_col = rearrange(video, 'c t h (p0 w) -> c t h p0 w', p0=fragment_num, w=self.fragment_size)
        part = fragment_col[:, :, :, 1::2, :]  # c t 3 32 w
        # c t 3 32 7 32
        part_row = rearrange(part, 'c t (p1 h) p0 w -> c t p1 h p0 w', p1=fragment_num, w=self.fragment_size)
        part_row = torch.flip(part_row, dims=[2])
        part = rearrange(part_row, 'c t p1 h p0 w -> c t (p1 h) p0 w')  # c t 3 32 224
        fragment_col[:, :, :, 1::2, :] = part
        video = rearrange(fragment_col, 'c t h p0 w -> c t h (p0 w)')

        fragment_row = rearrange(video, 'c t (p0 h) w -> c t p0 h w', p0=fragment_num, h=self.fragment_size)
        part = fragment_row[:, :, 1::2, :, :]  # c t 3 32 w
        # c t 3 32 7 32
        part_col = rearrange(part, 'c t p0 h (p1 w) -> c t p0 h p1 w', p1=fragment_num, h=self.fragment_size)
        part_col = torch.flip(part_col, dims=[-2])
        part = rearrange(part_col, 'c t p0 h p1 w -> c t p0 h (p1 w)')  # c t 3 32 224
        fragment_row[:, :, 1::2, :, :] = part
        video = rearrange(fragment_row, 'c t p0 h w -> c t (p0 h) w')

        return video
