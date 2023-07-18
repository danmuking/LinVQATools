from unittest import TestCase

import torch

from models.unknown_net import UnKnownNet


class TestUnKnownNet(TestCase):
    def test(self):
        model = UnKnownNet()
        x = torch.zeros(1,3,3,224,224)
        model(inputs=x)
