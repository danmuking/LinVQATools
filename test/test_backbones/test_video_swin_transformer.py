from unittest import TestCase

import torch

from models.backbones.swin_backbone import SwinTransformer3D


class TestVideoSwinTransformer(TestCase):
    def test(self):
        model = SwinTransformer3D(base_x_size=(16, 224, 224))
        x = torch.zeros((1,3,16,224,224))
        model(x)