import os
from collections import OrderedDict
from functools import reduce
from unittest import TestCase

import torch
from torch import nn

from models.faster_vqa import FasterVQA


class TestFasterVQA(TestCase):
    def test(self):
        os.chdir('../')
        model = FasterVQA(backbone_size='swin_tiny_grpb', backbone={"fragments": dict(window_size=(4, 4, 4))})
        # print(model.state_dict())
        # i_state_dict = model.state_dict()
        # t_state_dict = OrderedDict()
        # for key, value in i_state_dict.items():
        #     if 'vqa_head' in key:
        #         i_state_dict[key] = torch.ones_like(i_state_dict[key])
        #     t_state_dict[key] = i_state_dict[key]
        # model.load_state_dict(t_state_dict, strict=True)
        # print(model.state_dict())
        # model.eval()
        # m = nn.BatchNorm3d(64)
        # video = torch.zeros([2, 64, 16, 7, 7])
        # output = m(video)
        # print(output)
        video = torch.ones((2, 3, 32, 224, 224))
        scores = model(inputs=video, mode="predict", gt_label=torch.tensor(1))
        print(scores)
        # if len(scores) > 1:
        #     y_pred = reduce(lambda x, y: x + y, scores)
        # else:
        #     y_pred = scores[0]
        # y_pred = y_pred.mean((-3, -2, -1))
        # print(y_pred)
