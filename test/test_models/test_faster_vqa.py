import os
from collections import OrderedDict
from functools import reduce
from unittest import TestCase

import torch
from einops import rearrange
from torch import nn

from models.faster_vqa import FasterVQA


class TestFasterVQA(TestCase):
    def test(self):
        os.chdir('../../')
        model = FasterVQA(
            backbone='uniformerv1',
            vqa_head=dict(name='VQAHead',in_channels=512,drop_rate=0.8,fc_in=8*7*7),
            # load_path="/data/ly/code/LinVQATools/pretrained_weights/swin_tiny_patch244_window877_kinetics400_1k.pth"
        )
        video = torch.ones((2, 3, 16, 224, 224))
        scores = model(inputs=video, mode="loss", gt_label=torch.tensor(1))
        print(scores)
        # if len(scores) > 1:
        #     y_pred = reduce(lambda x, y: x + y, scores)
        # else:
        #     y_pred = scores[0]
        # y_pred = y_pred.mean((-3, -2, -1))
        # print(y_pred)

    def test_torch(self):
        x = [x for x in range(224*224)]
        x = torch.tensor(x).int()
        x = x.reshape(224,224)
        # print(x)
        x = rearrange(x, '(p0 h) (p1 w) -> (p0 p1) h w', p0=7, p1=7)
        print(x.shape)
        print(x[0,...])
