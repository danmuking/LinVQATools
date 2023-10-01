from unittest import TestCase

import torch
import torch.nn

from models.backbones.mvit import MViT


class TestMViT(TestCase):
    def test(self):
        model = MViT(arch='tiny')
        x = torch.zeros((2, 3, 16, 224, 224))
        y = model(x)
        print(y[0][0].shape)
    def test_load(self):
        path = '/data/ly/code/LinVQATools/pretrained_weights/MViTv2_S_16x4_k400_f302660347.pyth'
        # weight = torch.load(path)['model_state']
        # print(weight.keys())

        model = MViT(arch='small',path=path)
        # info = model.load_state_dict(weight,strict=False)
        # print(info)

    def test_torch(self):
        x = torch.zeros((2, 3, 16, 224, 224))
        conv = torch.nn.Conv3d(3,3,kernel_size=(2, 4, 4), stride=(2, 4, 4))
        y = conv(x)
        print(y.shape)
