import os

import cv2
import decord
import numpy as np
from decord import VideoReader

from utils.similar.video_similar import VideoSimilator

decord.bridge.set_bridge("torch")


class VideoExtractor:
    """
    抽取视频中不相似的区块
    """

    @staticmethod
    def extract(video_path):
        vreader = VideoReader(video_path)
        shape = vreader[0].shape
        video = np.zeros((len(vreader) // 2, *shape), dtype=np.uint8)
        for i in range(0, len(vreader), 2):
            video[i // 2] = vreader[i]
        # print(video.shape)
        video_list = []
        h_num = 28
        w_num = 56
        h_interval = shape[0] // h_num
        w_interval = shape[1] // w_num
        for i in range(h_num):
            for j in range(w_num):
                temp = []
                for k in range(len(vreader) // 2 // 8):
                    video_cube = video[k * 8:(k + 1) * 8, i * h_interval:i * h_interval + 32,
                                 j * w_interval:j * w_interval + 32, :]
                    result = False
                    for item in temp:
                        result = VideoSimilator.compare(video_cube, item)
                        if result:
                            break
                    if not result:
                        temp.append(video_cube)
                    if len(temp) == 0:
                        temp.append(video_cube)
                # print(len(temp))
                video_list += temp
        print(len(video_list))
        unique_list = [video_list[0]]
        for video1 in video_list:
            result = False
            for i in range(w_num*3):
                video2 = unique_list[len(unique_list)-i-1 if len(unique_list)-i-1>0 else len(unique_list)-1]
                result = VideoSimilator.compare(video1, video2)
                if result:
                    break
            if not result:
                unique_list.append(video1)
        print(len(unique_list))
        video_path = video_path.split("/")
        video_path.insert(3,'unique')
        video_path = os.path.join('/',*video_path)[:-4]
        print(video_path)
        for i,video in enumerate(unique_list):
            for j in range(len(video)):
                img_path = video_path+'/{}/{}.png'.format(i,j)
                makedir(img_path)
                img = cv2.cvtColor(video[j],cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_path,img)

def makedir(path: str):
    dir_path = os.path.dirname(path)
    if (os.path.exists(dir_path)):
        pass
    else:
        os.makedirs(dir_path)