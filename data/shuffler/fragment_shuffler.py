import random

import torch
from mmengine import MMLogger

from data import logger
from .base_shuffler import BaseShuffler
class FragmentShuffler(BaseShuffler):
    """
    从fragment cube上打乱数据
    """
    def __init__(self,fragment_size:int=32, frame_cube: int = 8,**kargs):
        self.fragment_size = fragment_size
        self.frame_cube = frame_cube
    def shuffle(self, video: torch.Tensor):
        c, t, h, w = video.shape
        logger.debug("从fragment cube维度打乱视频")
        matrix = []
        assert w % self.fragment_size == 0, '视频宽度无法被fragment_size整除,W:{} fragment_size:{}'.format(w, self.fragment_size)
        assert h % self.fragment_size == 0, '视频高度无法被fragment_size整除,H:{} fragment_size:{}'.format(h, self.fragment_size)
        assert t % self.frame_cube == 0, '视频帧数无法被frame_cube整除,T:{} frame_cube:{}'.format(t, self.frame_cube)
        num_cube = t // self.frame_cube
        num_w = w // self.fragment_size
        num_h = h // self.fragment_size
        for i in range(num_h):
            for j in range(num_w):
                for k in range(num_cube):
                    matrix.append((i, j, k))
        random.shuffle(matrix)
        count = 0
        target_video = torch.zeros_like(video)
        for i in range(num_w):
            for j in range(num_h):
                for k in range(num_cube):
                    h_s, h_e = i * self.fragment_size, (i + 1) * self.fragment_size
                    w_s, w_e = j * self.fragment_size, (j + 1) * self.fragment_size
                    t_s, t_e = k * self.frame_cube, (k + 1) * self.frame_cube
                    h_so, h_eo = matrix[count][0] * self.fragment_size, (matrix[count][0] + 1) * self.fragment_size
                    w_so, w_eo = matrix[count][1] * self.fragment_size, (matrix[count][1] + 1) * self.fragment_size
                    t_so, t_eo = matrix[count][2] * self.frame_cube, (matrix[count][2] + 1) * self.frame_cube
                    target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[
                                                                 :, t_so:t_eo, h_so:h_eo, w_so:w_eo
                                                                 ]
                    count = count + 1
        return target_video