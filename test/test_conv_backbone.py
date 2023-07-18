from unittest import TestCase

import torch

from models.backbones.conv_backbone import ConvNeXtV23D, convnextv2_tiny


class TestConvNeXtV23D(TestCase):
    def testConvNextXtV23D(self):
        model = convnextv2_tiny()
        x = torch.zeros(1, 3, 3, 224, 224,)
        feature = model(x)
        print([f.shape for f in feature])
