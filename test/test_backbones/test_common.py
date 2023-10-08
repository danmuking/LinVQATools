from unittest import TestCase

from models.utils.common import ChannelAttention


class TestChannelAttention(TestCase):
    def test(self):
        import torch
        x = torch.randn(2, 16, 16, 224, 224)
        model = ChannelAttention(16, 16)
        out = model(x)
        print(out.shape)
