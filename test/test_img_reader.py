import time
from unittest import TestCase

from data.file_reader.img_reader import ImgReader


class TestImgReader(TestCase):
    def test_read(self):
        reader = ImgReader('temp/fragment')
        start = time.time()
        reader.read('/data/ly/VQA_ODV/Group1/Reference/G1BikingToWork_3840x2160_fps23.976.mp4')
        print(time.time()-start)
