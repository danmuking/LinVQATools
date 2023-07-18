from unittest import TestCase

import torch

from models.unknown_net import UnknownNet


class TestUnknownNet(TestCase):
    def test(self):
        model = UnknownNet()
        r1 = torch.zeros([1, 3, 16, 224, 224])
        r2 = torch.zeros([1, 3, 4, 224, 224])
        x = [r1, r2]
        ans = model(inputs=x,mode='predict')
        print(ans)
