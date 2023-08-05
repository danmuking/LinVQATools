import random

import torch
from data import logger


class PostProcessSampler:
    """
    后处理采样，从已经预处理过的视频中抽取数据,只在时间维度上进行抽取
    """

    def __init__(
            self,
            frame_cube: int = 8,
            num=2,
            **kwargs
    ):
        self.frame_cube = frame_cube
        # 采样几个cube
        self.num = num

    def __call__(self, video, *args, **kwargs):
        logger.debug("进行数据后处理")
        c, t, h, w = video.shape
        assert t % self.frame_cube == 0, '视频帧数无法被frame_cube整除,T:{} frame_cube:{}'.format(t, self.frame_cube)
        num_cube = t // self.frame_cube
        index = random.sample(range(0, num_cube), self.num)
        target_video = torch.zeros((c, self.frame_cube * self.num, h, w))
        for i, item in enumerate(index):
            target_video[:, i * self.frame_cube:(i + 1) * self.frame_cube, :, :] = \
                video[:, item * self.frame_cube:(item + 1) * self.frame_cube, :, :]
        return target_video
