import os
from unittest import TestCase

import cv2
import decord
from decord import VideoReader

from utils.similar.video_extractor import VideoExtractor

decord.bridge.set_bridge("torch")


class TestVideoExtractor(TestCase):
    def test_extract(self):
        os.chdir('../')
        video_extractor = VideoExtractor().extract(
            '/data/ly/VQA_ODV/Group10/Reference/G10PandaBaseChengdu_7680x3840_fps29.97.mp4')

    def test(self):
        video_path = '/data/ly/VQA_ODV/Group10/Reference/G10PandaBaseChengdu_7680x3840_fps29.97.mp4'
        vreader = VideoReader(video_path)
        img = vreader[0].numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        print(img.shape)
        for i in range(14):
            cv2.line(img, (0, i * img.shape[0]//14), (img.shape[1], i * img.shape[0]//14), (255, 0, 255), 1)
        for i in range(28):
            cv2.line(img, (i * img.shape[1]//28, 0), (i * img.shape[1]//28, img.shape[0]), (255, 0, 255), 1)
        # cv2.rectangle(img, (0,0), (128,128), (255, 0, 255), 1)
        cv2.imwrite('out.png', img)
