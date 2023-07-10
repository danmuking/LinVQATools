from unittest import TestCase

import torch

from models.swin_backbone import SwinTransformer3D


class TestVideoSwinTransformer(TestCase):
    def test(self):
        model = SwinTransformer3D(base_x_size=(16,112,112))
        x = torch.zeros((1,3,16,112,112))
        ans = model(x)
        print(ans.shape)