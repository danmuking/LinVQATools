import os
from unittest import TestCase

import torch

from models.dover import DOVER, DoverWrapper


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
        model = DoverWrapper()
        views = dict()
        views['technical'] = torch.zeros((1, 3, 32, 224, 224))
        views['aesthetic'] = torch.zeros((1, 3, 32, 224, 224))
        scores = model(inputs=views, mode="loss", gt_label=torch.tensor(1))
        print(scores)
