import os
from unittest import TestCase

from data.default_dataset import DefaultDataset


class TestDefaultDataset(TestCase):
    def test_default_dataset(self):
        os.chdir('../')
        dataset = DefaultDataset(anno_reader='ODVVQAReader',split_file='./data/odv_vqa/tr_te_VQA_ODV.txt')
