from unittest import TestCase

from data.file_reader.image_reader import ImageReader


class TestImageReader(TestCase):
    def test(self):
        print(ImageReader.read('/data/ly/code/LinVQATools/test.png'))
