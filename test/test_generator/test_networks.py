from unittest import TestCase

import torch

from models.generator.networks import UpsamplingGenerator


class TestUpsamplingGenerator(TestCase):

    def test(self):
        model = UpsamplingGenerator(input_nc=768, output_nc=3, )
        x = torch.zeros((1, 768, 7, 7))
        y = model(x)
        print(y.shape)

    def test_forward(self):
        self.fail()
