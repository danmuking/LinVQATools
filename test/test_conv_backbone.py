from unittest import TestCase

import torch

from models.backbones.conv_backbone import ConvNeXtV23D, convnextv2_tiny


class TestConvNeXtV23D(TestCase):
    def testConvNextXtV23D(self):
        model = convnextv2_tiny()
        x = torch.zeros(1, 3, 3, 224, 224,)
        feature = model(x)
        print([f.shape for f in feature])
    def test_load_weight(self):
        model = convnextv2_tiny()
        path = '/home/ly/code/LinVQATools/pretrained_weights/convnextv2_tiny_22k_224_ema.pt'
        model.load_weight(path)
