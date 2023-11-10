import os
import random
from collections import OrderedDict
from functools import reduce
from unittest import TestCase

import torch
from einops import rearrange
from torch import nn

from models.backbones.video_mae_v2 import get_sinusoid_encoding_table
from models.faster_vqa import FasterVQA


class TestFasterVQA(TestCase):
    def test(self):
        os.chdir('../../')
        model = FasterVQA(
            backbone='vit',
            base_x_size=(16, 224, 224),
            vqa_head=dict(name='VQAHead',in_channels=384,drop_rate=0.8,fc_in=1568),
            load_path="/data/ly/code/LinVQATools/pretrained_weights/vit_s_k710_dl_from_giant.pth"
        )
        video = torch.ones((2, 3, 16, 224, 224))
        pos_embed = torch.ones((2,1568, 384))
        scores = model(inputs=video, mode="loss", gt_label=torch.tensor(1),pos_embed=pos_embed)
        print(scores)
        # if len(scores) > 1:
        #     y_pred = reduce(lambda x, y: x + y, scores)
        # else:
        #     y_pred = scores[0]
        # y_pred = y_pred.mean((-3, -2, -1))
        # print(y_pred)

    def test_torch(self):
        num_w = 7
        num_h = 7
        matrix = []
        for i in range(num_h):
            for j in range(num_w):
                matrix.append((i, j))
        random.shuffle(matrix)
        pos_embed = [x for x in range(1568)]
        pos_embed = torch.tensor(pos_embed).reshape(1, 1568, 1)
        pos_embed = rearrange(pos_embed, 'b (t h w) c -> (b c) t h w', t=8, h=14, w=14)
        target_pos_embed = torch.zeros_like(pos_embed)
        count = 0
        for i in range(num_w):
            for j in range(num_h):
                h_s, h_e = i * 2, (i + 1) * 2
                w_s, w_e = j * 2, (j + 1) * 2
                h_so, h_eo = matrix[count][0] * 2, (matrix[count][0] + 1) * 2
                w_so, w_eo = matrix[count][1] * 2, (matrix[count][1] + 1) * 2
                target_pos_embed[:, :, h_s:h_e, w_s:w_e] = pos_embed[
                                                           :, :, h_so:h_eo, w_so:w_eo
                                                           ]
                count = count + 1
        print(target_pos_embed.shape)
        print(matrix)
        print(target_pos_embed)