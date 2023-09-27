import os
from unittest import TestCase

from torch.utils.data import DataLoader
from tqdm import tqdm

from data.image_dataset import ImageDataset


class TestImageDataset(TestCase):
    def test(self):
        os.chdir('../../')
        dataset = ImageDataset(norm=False)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        for i in tqdm(dataloader):
            # print(i)
            pass
        # for i in range(len(dataset)):
        #     dataset.__getitem__(i)
