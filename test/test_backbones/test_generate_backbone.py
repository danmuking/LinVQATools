from unittest import TestCase

import torch

from models.backbones.generate_backbone import SwinTransformer3D


class TestGenerateBackbone(TestCase):
    def test(self):
        model = SwinTransformer3D(base_x_size=(16, 224, 224), )
        x = torch.zeros((2, 3, 16, 224, 224))
        y = model(x)
        vqa_feat = y[0][0]
        generate_feat = y[0][1]
        print(vqa_feat.shape,generate_feat.shape)
