import os
from functools import reduce
from unittest import TestCase

import torch

from models.faster_vqa import FasterVQA


class TestFasterVQA(TestCase):
    def test(self):
        os.chdir('../')
        model = FasterVQA(backbone_size='swin_tiny_grpb',backbone={"fragments": dict(window_size=(4, 4, 4))},load_path="./pretrained_weights/FAST_VQA_3D_1_1.pth")
        video = torch.randn((1, 3, 32, 224, 224))
        video = dict(fragments=video)
        scores = model(video,mode="loss")
        if len(scores) > 1:
            y_pred = reduce(lambda x, y: x + y, scores)
        else:
            y_pred = scores[0]
        y_pred = y_pred.mean((-3, -2, -1))
        print(y_pred)
