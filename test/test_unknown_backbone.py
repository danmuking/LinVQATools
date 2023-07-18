from unittest import TestCase

import torch

from models.backbones.unknown_backbone import UnKnownNet, FusionBlock


class TestUnKnownNet(TestCase):
    def test(self):
        model = UnKnownNet()
        r1 = torch.zeros([1, 3, 16, 224, 224])
        r2 = torch.zeros([1, 3, 4, 224, 224])
        x = [r1, r2]
        model(inputs=x)


class TestFusionBlock(TestCase):
    def test(self):
        r1 = torch.zeros([1, 384, 8, 14, 14])
        r2 = torch.zeros([1, 384, 3, 14, 14])
        r3 = torch.zeros([1, 384, 11, 14, 14])
        model = FusionBlock(384, 2 * 384)
        res = model(r1=r1, r2=r2, r3=r3)
        print(res.shape)
