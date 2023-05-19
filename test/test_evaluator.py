import os
from functools import reduce
from unittest import TestCase

import torch

from models.evaluators import DiViDeAddEvaluator


class TestDiViDeAddEvaluator(TestCase):
    def test1(self):
        os.chdir('../')
        model = DiViDeAddEvaluator(backbone_size='swin_tiny_grpb',backbone={"fragments": dict(window_size=(4, 4, 4))})
        video = torch.randn((1,3,32,224,224))
        video = dict(fragments=video)
        scores = model(video)
        if len(scores) > 1:
            y_pred = reduce(lambda x, y: x + y, scores)
        else:
            y_pred = scores[0]
        y_pred = y_pred.mean((-3, -2, -1))
        print(y_pred)
