import time
from unittest import TestCase

from data.file_reader.img_reader import ImgReader
from data.file_reader.random_img_reader import RandomImgReader


class TestImgReader(TestCase):
    def test_img_read(self):
        reader = ImgReader('temp/fragment')
        start = time.time()
        data = reader.read('/data/ly/VQA_ODV/Group1/G1AbandonedKingdom_ERP_7680x3840_fps30_qp27_45406k.mp4',cube_num=2)
        print(data.shape)
        print(time.time()-start)
    def test_random_img_read(self):
        reader = RandomImgReader('resize')
        start = time.time()
        data = reader.read('/data/ly/VQA_ODV/Group1/G1AbandonedKingdom_ERP_7680x3840_fps30_qp27_45406k.mp4',
                    frame_num=3)
        print(data.shape)
        print(time.time()-start)
