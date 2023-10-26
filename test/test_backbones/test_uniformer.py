from unittest import TestCase

import torch

from models.backbones.uniformerv1 import uniformer_x, conv_3xnxn


# from models.backbones.swin_backbone import SwinTransformer3D


class TestUniformer(TestCase):
    def test(self):
        model = uniformer_x()
        x = torch.zeros((2, 3, 16, 224, 224))
        y = model(x)
        path = "/data/ly/code/LinVQATools/pretrained_weights/uniformer_small_k600_16x4.pth"
        weight = torch.load(path)
        info = model.load_state_dict(weight,strict=False)
        print(y.shape)
        print(info)

    def test_load(self):
        path = "/data/ly/code/LinVQATools/pretrained_weights/uniformer_small_k600_16x4.pth"
        weight = torch.load(path)
        print(weight.keys())