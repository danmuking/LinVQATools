import random

import torch
from mmengine import MMLogger

from data import logger
from .base_shuffler import BaseShuffler
class SpatialShuffler(BaseShuffler):
    """
    实现数据空间维度上的打乱
    """
    def __init__(self, fragment_size: int = 32, **kargs):
        super().__init__(**kargs)
        self.fragment_size = fragment_size

    def __call__(self,video:torch.Tensor):
        c,t,h,w = video.shape
        logger.debug("从空间维度打乱视频")
        matrix = []
        assert w % self.fragment_size == 0, '视频宽度无法被fragment_size整除,W:{} fragment_size:{}'.format(w, self.fragment_size)
        assert h % self.fragment_size == 0, '视频高度无法被fragment_size整除,H:{} fragment_size:{}'.format(h, self.fragment_size)
        num_w = w // self.fragment_size
        num_h =  h//self.fragment_size
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
        return target_video
