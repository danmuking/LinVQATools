import os
from unittest import TestCase

import torch

from models.backbone.conv_backbone import convnext_3d_tiny
from models.dover import DOVER, DoverWrapper
from models.backbone.swin_backbone import SwinTransformer3D as VideoBackbone

class TestDOVER(TestCase):
    def testDover(self):
        os.chdir('../')
        args = {'backbone': {'technical': {'type': 'swin_tiny_grpb', 'checkpoint': True, 'pretrained': None},
                             'aesthetic': {'type': 'conv_tiny'}}, 'backbone_preserve_keys': 'technical,aesthetic',
                'divide_head': True, 'vqa_head': {'in_channels': 768, 'hidden_channels': 64}}
        evaluator = DOVER(**args)
        views = dict()
        views['technical'] = torch.zeros((1, 3, 32, 224, 224))
        views['aesthetic'] = torch.zeros((1, 3, 32, 224, 224))
        result = evaluator(views)
        print(result)

    def testDoverWrapper(self):
        # model =
        views = dict()
        model = DoverWrapper()
        views['technical'] = torch.zeros((2, 3, 32, 224, 224))
        views['aesthetic'] = torch.zeros((2, 3, 32, 224, 224))
        scores = model(inputs=views, mode="predict", gt_label=torch.tensor(1))
        # x = torch.zeros((1, 3, 32, 224, 224))
        # a = model(x)
        # print(a.shape)
        # model = convnext_3d_tiny(pretrained=True)
        # b = torch.zeros((1, 3, 32, 224, 224))
        # b = model(x)
        # print(b.shape)
        # print(torch.cat((a,b),dim=1).shape)
        # print(model.state_dict().keys())
        print(scores)
