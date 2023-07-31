import os
from unittest import TestCase

import torch

from models.biformer_arc import BiFormerArc


class TestBiFormerArc(TestCase):
    def test_model(self):
        os.chdir('../../')
        model = BiFormerArc(load_path='./pretrained_weights/biformer_tiny_best.pth')
        video = torch.ones((2, 3, 32, 224, 224))
        scores = model(inputs=video, mode="loss", gt_label=torch.tensor((2,1)))
        print(scores)
