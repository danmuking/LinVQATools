import time

import torch
import cv2
import decord
import numpy
import numpy as np
from decord import VideoReader

from utils.similar.img_similar import ImgSimilator

decord.bridge.set_bridge("torch")


class VideoSimilator:
    @staticmethod
    def hanming_dist(s1, s2):
        """
        求汉明距离
        """
        # print(s1, s2)
        return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])

    @staticmethod
    def compare(video1, video2) -> bool:
        """
        输入比较的两个视频
        video [frame_num,H,W,C]
        返回是否相似，帧相似度>=0.85直接返回True，否则返回False
        """

        # 获取较短视频的帧数
        min_frame_count = min(len(video1),
                              len(video1))

        # 获取视频FPS
        fps = 1

        similar = 0
        frame_cnt = int(min_frame_count / fps)
        # 截帧
        for i in range(frame_cnt):
            frame1 = video1[i]
            frame2 = video2[i]
            phash1 = ImgSimilator.pHash(frame1)
            phash2 = ImgSimilator.pHash(frame2)

            # 比较汉明距离
            if VideoSimilator.hanming_dist(phash1, phash2) < 18:
                similar += 1
        return similar / frame_cnt == 1


if __name__ == '__main__':
    video_path = '/data/ly/VQA_ODV/Group1/Reference/G1BikingToWork_3840x2160_fps23.976.mp4'
    vreader = VideoReader(video_path)
    shape = vreader[0].shape
    video = np.zeros((len(vreader)//2,*shape),dtype=np.uint8)
    for i in range(0,len(vreader),2):
        video[i//2] = vreader[i]
    # video = torch.stack(imgs, 0)
    video1 = video[:8,0:32,0:32,:]
    video2 = video[:8,128:160,1:33,:]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 设置视频帧频
    fps = 10
    # 设置视频大小
    size = (32, 32)
    out = cv2.VideoWriter('./out1.avi', fourcc, fps, size)
    for i in range(video1.shape[0]):
        fra = video1[i]
        print(fra)
        print(fra.shape)
        # fra = fra.permute(1, 2, 0)
        fra = cv2.cvtColor(fra, cv2.COLOR_RGB2BGR)
        out.write(fra)
    out.release()
    out = cv2.VideoWriter('./out2.avi', fourcc, fps, size)
    for i in range(video2.shape[0]):
        fra = video2[i]
        # fra = fra.permute(1, 2, 0)
        # fra = fra.numpy().astype(np.uint8)
        fra = cv2.cvtColor(fra, cv2.COLOR_RGB2BGR)
        out.write(fra)
    out.release()

    # video1 = np.zeros((8, 32, 32, 3),dtype=np.uint8)
    # video2 = np.ones((8, 32, 32, 3),dtype=np.uint8)
    print(VideoSimilator.compare(video1,video2))
