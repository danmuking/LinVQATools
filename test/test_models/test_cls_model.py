import os
from unittest import TestCase

import torch

from models.cls_model import ClsModel


class TestClsModel(TestCase):
    def test(self):
        os.chdir('../../')
        model = ClsModel(load_path="/data/ly/code/LinVQATools/pretrained_weights/vit_s_k710_dl_from_giant.pth")
        video = torch.ones((2, 3, 16, 224, 224))
        scores = model(inputs=video, mode="loss", gt_label=torch.tensor(1), sort_list=torch.zeros((2, 49)),
                       y_c=torch.zeros((2, 49)), y_r=torch.zeros((2, 49)),grade=torch.ones((2,1)))
        print(scores)
