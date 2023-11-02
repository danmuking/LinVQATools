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
            backbone='vit',
            base_x_size=(16, 224, 224),
            vqa_head=dict(name='VQAHead',in_channels=384,drop_rate=0.8,fc_in=1568),
            load_path="/data/ly/code/LinVQATools/pretrained_weights/vit_b_k710_dl_from_giant.pth"
        )
        video = torch.ones((2, 3, 32, 224, 224))
        scores = model(inputs=video, mode="loss", gt_label=torch.tensor(1))
        print(scores)
        # if len(scores) > 1:
        #     y_pred = reduce(lambda x, y: x + y, scores)
        # else:
        #     y_pred = scores[0]
        # y_pred = y_pred.mean((-3, -2, -1))
        # print(y_pred)

    def test_torch(self):
        rnn = nn.GRU(2048, 2048, 2)
        input = torch.randn(3, 8, 2048)
        output, hn = rnn(input)
        print(output.shape)
        print(hn.shape)