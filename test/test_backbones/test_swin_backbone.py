from unittest import TestCase

import torch

from models.backbones.swin_backbone import SwinTransformer3D


class TestSwinTransformer3D(TestCase):
    def test_load(self):
        path = '/data/ly/code/LinVQATools/pretrained_weights/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-241016b2.pth'
        weight = torch.load(path)
        print(weight.keys())
        model = SwinTransformer3D(base_x_size=(16, 224, 224),
                                  load_path=path)
        # print(weight['model'].keys())

        # t_state_dict = model.state_dict()
        # print(t_state_dict.keys())
        # s_state_dict = torch.load(path)['state_dict']
        # print(s_state_dict.keys())
