import os
from unittest import TestCase

from data.split.dataset_split import DatasetSplit


class TestDatasetSplit(TestCase):
    def test_split(self):
        os.chdir('../')
        test_data = []
        for i in range(100):
            test_data.append(dict(
                        scene_id=i
            ))
        print(DatasetSplit.split(test_data))
        print(DatasetSplit.split(test_data,'./data/odv_vqa/tr_te_VQA_ODV.txt'))
