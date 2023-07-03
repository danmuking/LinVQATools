import os
from unittest import TestCase

from data.mnist_dataset import MnistDatasetWrapper


class TestMnistDatasetWrapper(TestCase):
    def testMnistDatasetWrapper(self):
        os.chdir('../')
        dataset = MnistDatasetWrapper()
        print(dataset[0])
