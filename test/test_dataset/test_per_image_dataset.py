import os
from unittest import TestCase

import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.per_image_dataset import PerImageDataset


class TestPerImageDataset(TestCase):
    def test(self):
        os.chdir('../../')
        prefix = 'temp/fragment'
        dataset = PerImageDataset(
            prefix=prefix,
            anno_reader='ODVVQAReader',
            split_file='./data/odv_vqa/tr_te_VQA_ODV.txt',
            phase='test',
        )
        dataloader = DataLoader(dataset, batch_size=1, num_workers=4,shuffle=True)
        for item in tqdm(dataloader):
            # print(item)
            pass

    def test_save_img(self):
        os.chdir('../../')
        prefix = 'temp/fragment'
        dataset = PerImageDataset(
            prefix=prefix,
            anno_reader='ODVVQAReader',
            split_file='./data/odv_vqa/tr_te_VQA_ODV.txt',
            phase='train',
        )
        img = dataset[0]['inputs']
        fra = img.permute(1, 2, 0)
        fra = fra.numpy().astype(np.uint8)
        fra = cv2.cvtColor(fra, cv2.COLOR_RGB2BGR)
        cv2.imwrite('test.png',fra)
        print(dataset[0]['inputs'].shape)
