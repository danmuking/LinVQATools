from unittest import TestCase

import torch

from models.unknown_net import UnknownNet


class TestUnknownNet(TestCase):
    def test(self):
        model = UnknownNet()
        r1 = torch.zeros([2, 3, 16, 224, 224])
        r2 = torch.zeros([2, 3, 4, 224, 224])
        x = [r1, r2]
        print(model.state_dict().keys())
        ans = model(inputs=x, mode='loss',gt_label=torch.tensor([1,1]))
        print(ans)
