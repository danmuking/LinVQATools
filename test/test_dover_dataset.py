import os
from unittest import TestCase

from torch.utils.data import DataLoader

from data.dover_dataset import DoverDataset


class TestDoverDataset(TestCase):
    def testDoverDataset(self):
        os.chdir('../')
        frame_sampler = dict(
            name='FragmentSampleFrames',
            fsize_t=32 // 8,
            fragments_t=8,
            clip_len=32,
            frame_interval=2,
            t_frag=8,
            num_clips=1,
        )
        spatial_sampler = dict(
            name='PlaneSpatialFragmentSampler',
            fragments_h=7,
            fragments_w=7,
            fsize_h=32,
            fsize_w=32,
            aligned=8,
        )
        dataset = DoverDataset(anno_reader='ODVVQAReader', split_file='./data/odv_vqa/tr_te_VQA_ODV.txt',
                                 frame_sampler=frame_sampler, spatial_sampler=spatial_sampler, prefix='fragment')
        dataloader = DataLoader(dataset, batch_size=6, num_workers=4)
        # for item in tqdm(dataloader):
        #     print(item)
        dataset[0]
