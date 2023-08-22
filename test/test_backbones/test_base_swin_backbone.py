from unittest import TestCase

import torch

from models.backbones.base_swin_backbone import SwinTransformer3D


class TestSwinTransformer3D(TestCase):
    def test(self):
        model = SwinTransformer3D(arch='tiny')
        x = torch.zeros(2, 3, 32, 224, 224)
        print(model(x))

    def test_load(self):
        path = '/data/ly/code/LinVQATools/pretrained_weights/swin_tiny_patch244_window877_kinetics400_1k.pth'
        weight = torch.load(path)
        model = SwinTransformer3D(arch='tiny')
        # print(weight['model'].keys())

        t_state_dict = model.state_dict()
        print(t_state_dict.keys())
        s_state_dict = torch.load(path)['state_dict']
        print(s_state_dict.keys())
        # from collections import OrderedDict
        # for key in t_state_dict.keys():
        #     if key not in s_state_dict:
        #         print(key)
        #         continue
        #     if t_state_dict[key].shape != s_state_dict[key].shape:
        #         print(t_state_dict[key].shape, s_state_dict[key].shape)
        #         t = t_state_dict[key].shape[2]
        #         s_state_dict[key] = s_state_dict[key].unsqueeze(2).repeat(1, 1, t, 1, 1) / t
        # info = model.load_state_dict(s_state_dict, strict=False)
        # print(info)
