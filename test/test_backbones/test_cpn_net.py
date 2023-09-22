
from unittest import TestCase
import torch

from models.backbones.cpn_net import CPNet

class TestCPNet(TestCase):
    def test(self, args=None):

        x = torch.randn((2, 3, 224, 224))
        net = CPNet(args)
        y = net(x)

        print(y.shape)
