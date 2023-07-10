"""
从一个个图片切片中读取视频
"""
import os
import random

import cv2
import torch
from mmengine import MMLogger

logger = MMLogger.get_instance('dataset')

class ClipReader:
    def __init__(self,*args,**kwargs):
        logger.info("ClipReader:无效参数{},{}".format(args,kwargs))
    def read(self, video_path: str,*args,**kwargs) -> torch.Tensor:
        """
        读取视频
        :param video_path: 原始视频路径
        :param is_train: 是否为训练模式
        :return: 成功返回视频，失败返回none
        """

        # 预处理好的视频路径
        video_pre_path = video_path.split('/')
        video_pre_path.insert(3, 'unique')
        video_pre_path = os.path.join('/', *video_pre_path)[:-4]
        # print(video_pre_path)
        video_cube_list = [i for i in range(len(os.listdir(video_pre_path)))]
        random.shuffle(video_cube_list)
        video_cube_list = video_cube_list[:7*7*4]
        video = torch.zeros((32,224,224,3))
        for i in range(7):
            for j in range(7):
                for k in range(4):
                    num = i*28+j*4+k
                    video_cube_path = os.path.join(video_pre_path,str(video_cube_list[num]))
                    video_cube = []
                    for x in range(8):
                        video_cube_img_path = os.path.join(video_cube_path,'{}.png'.format(x))
                        # print(video_cube_img_path)
                        frame = cv2.imread(video_cube_img_path)
                        # print(frame)
                        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                        video_cube.append(torch.tensor(frame))
                    video_cube = torch.stack(video_cube,dim=0)
                    video[8*k:8*(k+1),32*i:32*(i+1),32*j:32*(j+1),:] = video_cube
        video = video.permute(3, 0, 1, 2)

        return video
