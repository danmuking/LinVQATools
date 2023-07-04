"""
实现图片相似度计算
"""
import cv2
import numpy as np


class ImgSimilator:
    @staticmethod
    def pHash(img):
        """
        get image pHash value
        """

        # 缩放图片为32x32灰度图片
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)

        # 创建二维列表
        h, w = img.shape[:2]
        vis0 = np.zeros((h, w), np.float32)
        vis0[:h, :w] = img

        # 二维Dct变换
        vis1 = cv2.dct(cv2.dct(vis0))
        vis1 = vis1[:8, :8]

        # 把二维list变成一维list
        img_list = vis1.flatten().tolist()

        # 计算均值, 得到哈希值
        avg = sum(img_list) * 1. / 64
        avg_list = [0 if i < avg else 1 for i in img_list]
        return avg_list
