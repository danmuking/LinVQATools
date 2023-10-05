from unittest import TestCase

import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from models.backbones.biformer.ops.bra_legacy import BiLevelRoutingAttention, TopkRouting, KVGather, QKVLinear
from models.backbones.swin_backbone import SwinTransformer3D, SwinTransformerBlock3D, BasicLayer


class TestSwinTransformer3D(TestCase):
    def test(self):
        model = SwinTransformer3D(base_x_size=(16, 224, 224), window_size=(4, 7, 7), )
        x = torch.zeros((2, 3, 16, 224, 224))
        y = model(x)
        print(y[0][0].shape)

    def test_BiLevelRoutingAttention(self):
        x = torch.ones((3,8, 7, 7, 16))
        atte = BiLevelRoutingAttention(dim=16)
        y = atte(x)
        print(y.shape)

    def test_swin_block(self):
        x = torch.ones((2, 3, 16, 224, 224))
        block  = SwinTransformer3D()
        y = block(x)
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
