from unittest import TestCase

import numpy as np
import torch

from models.backbones.swin_backbone import SwinTransformer3D, global_position_index


class TestSwinTransformer3D(TestCase):
    def test(self):
        model = SwinTransformer3D(base_x_size=(16, 224, 224), window_size=(4, 7, 7), )
        x = torch.zeros((2, 3, 16, 224, 224))
        y = model(x)
        print(y[0][0].shape)

    def test_load(self):
        path = '/data/ly/code/LinVQATools/pretrained_weights/swin_tiny_patch244_window877_kinetics400_1k.pth'
        weight = torch.load(path)['state_dict']
        # print(weight.keys())

        model = SwinTransformer3D(base_x_size=(16, 224, 224),
                                  load_path=path)
        # print(weight['model'].keys())

        # t_state_dict = model.state_dict()
        # print(t_state_dict.keys())
        # s_state_dict = torch.load(path)['state_dict']
        # print(s_state_dict.keys())

    def test_gpi(self):
        gpi = global_position_index(8, 7, 7, fragments=(2, 7, 7), window_size=(8, 7, 7), )
        print(gpi.shape)
        torch.set_printoptions(threshold=np.inf)
        # print(gpi[0,0,...])
        print(gpi.abs().sum(-1).shape)
        print(gpi.abs().sum(-1)[0,0].reshape(8,7,7))
