import os
from collections import OrderedDict
from functools import reduce
from unittest import TestCase

import torch
from torch import nn

from models.faster_vqa import FasterVQA


class TestFasterVQA(TestCase):
    def test(self):
        os.chdir('../../')
        model = FasterVQA(
            backbone='faster_vqa',
            base_x_size=(16, 224, 224),
            vqa_head=dict(name='VQAHead', in_channels=768, fc_in=1 * 7 * 7),
            # load_path="/data/ly/code/LinVQATools/pretrained_weights/MViTv2_S_16x4_k400_f302660347.pyth"
        )
        video = torch.ones((2, 3, 16, 224, 224))
        scores = model(inputs=video, mode="loss", gt_label=torch.tensor((2,1)),camera_motion=torch.zeros((2,1)))
        print(scores)
        # if len(scores) > 1:
        #     y_pred = reduce(lambda x, y: x + y, scores)
        # else:
        #     y_pred = scores[0]
        # y_pred = y_pred.mean((-3, -2, -1))
        # print(y_pred)

    def test_torch(self):
        loss = nn.CrossEntropyLoss()
        input = torch.randn(3, 5, requires_grad=True)
        target = torch.empty(3, dtype=torch.long).random_(5)
        print(input,target)
        output = loss(input, target)
        output.backward()