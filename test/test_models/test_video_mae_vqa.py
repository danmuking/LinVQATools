from collections import OrderedDict
from unittest import TestCase

import numpy as np
import torch
from einops import rearrange

from models.video_mae_vqa import VideoMAEVQA, VideoMAEVQAWrapper, RandomCellMaskAgent


class TestVideoMAEVQA(TestCase):
    def test(self):
        model = VideoMAEVQA()
        x = {'video': torch.rand((2, 3, 16, 224, 224)), "mask": torch.ones((2, 8, 196)).long()}
        y = model(x)
        print(y)

    def test_mask(self):
        # torch.set_printoptions(threshold=np.inf)
        # model = CellRunningMaskAgent()
        # x = {'video': torch.rand((2, 3, 16, 224, 224)), "mask": torch.zeros((2, 8 * 14 * 14)).long()}
        # y = model(x, [8, 14, 14])
        # mask = y['mask']
        # print(mask.shape)
        # mask = mask.reshape(mask.size(0), 8, -1)
        # print(mask.shape)
        # print(mask[0].reshape(8,14,14))

        torch.set_printoptions(threshold=np.inf)
        # model = BlockMaskAgent()
        model = RandomCellMaskAgent()
        x = {'video': torch.rand((2, 3, 16, 224, 224)), "mask": torch.zeros((2, 8 * 14 * 14)).long()}
        y = model(x, [8, 14, 14])
        mask = y['mask']
        print(mask.shape)
        mask = mask.reshape(mask.size(0), 8, -1)
        print(mask.shape)
        print(mask[0].reshape(8, 14, 14))

    def test_VideoMAEVQAWrapper(self):
        model = VideoMAEVQAWrapper()
        y = model(inputs=torch.rand((1,4, 3, 16, 224, 224)), gt_label=torch.rand((2)),mode='predict')
        print(y)

    def test_load(self):
        # weight = torch.load("/data/ly/code/LinVQATools/pretrained_weights/vit_s_k710_dl_from_giant.pth")
        # print(weight.keys())
        # print(weight['module'].keys())
        # weight = weight['module']
        t_state_dict = OrderedDict()
        # for key in weight.keys():
        #     weight_value = weight[key]
        #     key = "model.backbone." + key
        #     # if 'encoder' in key:
        #     #     key = key.replace('encoder', 'backbone')
        #     t_state_dict[key] = weight_value
        # weight = torch.load("/data/ly/code/LinVQATools/pretrained_weights/video_mae_k400.pth")
        weight = torch.load("/data/ly/code/LinVQATools/pretrained_weights/video_mae_v1_s_pretrain.pth")
        print(weight['model'].keys())
        weight = weight['model']
        for key in weight.keys():
            if "decoder" in key:
                weight_value = weight[key]
                key = "model." + key
                t_state_dict[key] = weight_value
        t_state_dict = OrderedDict(filter(lambda x: 'encoder_to_decoder' not in x[0], t_state_dict.items()))
        print(t_state_dict.keys())
        model = VideoMAEVQAWrapper()
        info = model.load_state_dict(t_state_dict, strict=False)
        print(info)

    def test(self):
        x = torch.tensor([i for i in range(12)])
        x = rearrange(x,"(r w) -> r w",r=3,w=4)
        print(x)
