import random

import torch

from .base_shuffler import BaseShuffler
from data import logger
class TimeShuffler(BaseShuffler):
    def __int__(self, frame_cube: int = 8,**kargs):
        self.frame_cube = frame_cube
    def shuffle(self, video: torch.Tensor):
        """
        实现数据时间维度上的打乱
        """

        c, t, h, w = video.shape
        logger.debug("从时间维度打乱视频,frame_cube:{}".format(self.frame_cube))
        matrix = []
        assert t % self.frame_cube == 0, '视频帧数无法被frame_cube整除,T:{} frame_cube:{}'.format(t,self.frame_cube)
        num_cube = t // self.frame_cube
        for i in range(num_cube):
                matrix.append(i)
        random.shuffle(matrix)
        count = 0
        target_video = torch.zeros_like(video)
        for k in range(num_cube):
            t_s, t_e = k * self.frame_cube, (k + 1) * self.frame_cube
            t_so, t_eo = matrix[count] * self.frame_cube, (matrix[count] + 1) * self.frame_cube
            target_video[:, t_s:t_e, :, :] = video[
                                                         :, t_so:t_eo, :, :
                                                         ]
            count = count + 1
        return target_video