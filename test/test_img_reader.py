import time
from unittest import TestCase

import cv2

from data.file_reader.img_reader import ImgReader


class TestImgReader(TestCase):
    def test_read(self):
        reader = ImgReader('temp/fragment')
        start = time.time()
        reader.read('/data/ly/temp/fragment/106/VQA_ODV/Group4/G4CliffsideMansion_RCMP_7680x3840_fps30_qp42_1286k_ERP.mp4')
        print(time.time()-start)
    def test(self):
        img = cv2.imread('/data/ly/temp/fragment/106/VQA_ODV/Group4/G4CliffsideMansion_RCMP_7680x3840_fps30_qp42_1286k_ERP/0.png')
        print(img)
