import os
from collections import OrderedDict
from functools import reduce
from unittest import TestCase

import torch
from torch import nn

from models.backbones.swin_backbone import global_position_index
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
        scores = model(inputs=video, mode="loss", gt_label=torch.tensor((2, 1)), camera_motion=torch.zeros((2, 1)))
        print(scores)
        # if len(scores) > 1:
        #     y_pred = reduce(lambda x, y: x + y, scores)
        # else:
        #     y_pred = scores[0]
        # y_pred = y_pred.mean((-3, -2, -1))
        # print(y_pred)

    def test_torch(self):
        x = torch.ones([2, 8, 56, 56, 96])
        _, Dp, Hp, Wp, _ = x.shape
        window_size = (8, 7, 7)
        shift_size = (0, 0, 0)
        gpi = global_position_index(
            Dp, Hp, Wp, fragments=(1,) + window_size[1:], window_size=window_size, shift_size=shift_size,
            device=x.device,
        )
        # print(gpi)
