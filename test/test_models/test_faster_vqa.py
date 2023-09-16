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
            vqa_head=dict(name='FcHead', in_channels=8 * 7 * 7),
            load_path="/data/ly/code/LinVQATools/pretrained_weights/MViTv2_S_16x4_k400_f302660347.pyth"
        )
        video = torch.ones((2, 3, 16, 224, 224))
        scores = model(inputs=video, mode="predict", gt_label=torch.tensor(1))
        print(scores)
        # if len(scores) > 1:
        #     y_pred = reduce(lambda x, y: x + y, scores)
        # else:
        #     y_pred = scores[0]
        # y_pred = y_pred.mean((-3, -2, -1))
        # print(y_pred)

    def test_torch(self):
        window_size = (2,2,2)
        shift_size = (2,2,2)
        D,H,W=4,4,4
        img_mask = torch.zeros((1, D, H, W, 1))  # 1 Dp Hp Wp 1
        cnt = 0
        for d in (
                slice(-window_size[0]),
                slice(-window_size[0], -shift_size[0]),
                slice(-shift_size[0], None),
        ):
            for h in (
                    slice(-window_size[1]),
                    slice(-window_size[1], -shift_size[1]),
                    slice(-shift_size[1], None),
            ):
                for w in (
                        slice(-window_size[2]),
                        slice(-window_size[2], -shift_size[2]),
                        slice(-shift_size[2], None),
                ):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
        x = [i for i in range(4)]
        test = slice(-window_size[0])
        print(test.start,test.stop,test.step)
        print(x[test])
        # print(img_mask.view(D, H, W))