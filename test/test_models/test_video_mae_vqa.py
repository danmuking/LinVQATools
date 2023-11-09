from collections import OrderedDict
from unittest import TestCase

import torch
from einops import rearrange
from torch import nn

from models.video_mae_vqa import VideoMAEVQA, CellRunningMaskAgent, VideoMAEVQAWrapper


class TestVideoMAEVQA(TestCase):
    def test(self):
        model = VideoMAEVQA()
        x = {'video': torch.rand((2, 3, 16, 224, 224)), "mask": torch.ones((2, 8, 196)).long()}
        y = model(x)
        print(y)

    def test_mask(self):
        model = CellRunningMaskAgent()
        x = {'video': torch.rand((2, 3, 16, 224, 224)), "mask": torch.zeros((2, 8 * 14 * 14)).long()}
        y = model(x, [8, 14, 14])
        mask = y['mask']
        mask = mask.reshape(mask.size(0), 8, -1)
        print(mask.shape)

    def test_VideoMAEVQAWrapper(self):
        model = VideoMAEVQAWrapper()
        y = model(inputs=torch.rand((2, 3, 16, 224, 224)), gt_label=torch.rand((2)),mode='loss')
        print(y)

    def test_load(self):
        weight = torch.load("/data/ly/code/LinVQATools/pretrained_weights/vit_b_k710_dl_from_giant.pth")
        print(weight.keys())
        print(weight['module'].keys())
        weight = weight['module']
        t_state_dict = OrderedDict()
        for key in weight.keys():
            weight_value = weight[key]
            key = "model.backbone." + key
            # if 'encoder' in key:
            #     key = key.replace('encoder', 'backbone')
            t_state_dict[key] = weight_value
        print(t_state_dict.keys())
        model = VideoMAEVQAWrapper()
        info = model.load_state_dict(t_state_dict, strict=False)
        print(info)
