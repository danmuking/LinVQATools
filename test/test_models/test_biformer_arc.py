from unittest import TestCase

import torch

from models.biformer_arc import BiFormerArc


class TestBiFormerArc(TestCase):
    def test_model(self):
        model = BiFormerArc(load_path='/data/ly/code/LinVQATools/pretrained_weights/biformer_tiny_best.pth')
        video = torch.ones((2, 3, 32, 224, 224))
        scores = model(inputs=video, mode="predict", gt_label=torch.tensor(1))
        print(scores)
