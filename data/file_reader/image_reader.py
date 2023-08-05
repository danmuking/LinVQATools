# 加载一张图片
import os

import cv2
import torch

from data import logger


class ImageReader:
    @staticmethod
    def read(img_path) -> torch.Tensor:
        """
        加载单帧图片,返回rgb格式图片
        Args:
            img_path:
        Returns:

        """
        if not os.path.exists(img_path):
            logger.error("{}加载失败".format(img_path))
            return None
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.tensor(img)
