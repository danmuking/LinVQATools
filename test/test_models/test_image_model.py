import os
from unittest import TestCase

import torch

from models.image_model import ImageModel


class TestImageModel(TestCase):
    def test(self):
        os.chdir('../../')
        model = ImageModel(
            backbone='faster_vqa',
            base_x_size=(16, 224, 224),
            vqa_head=dict(name='MeanHead'),
            load_path="/data/ly/code/LinVQATools/pretrained_weights/MViTv2_S_16x4_k400_f302660347.pyth"
        )
        video = torch.ones((2, 3, 320, 320))
        scores = model(inputs=video, mode="loss", gt_label=torch.tensor((2,1)))
        print(scores)
        # if len(scores) > 1:
        #     y_pred = reduce(lambda x, y: x + y, scores)
        # else:
        #     y_pred = scores[0]
        # y_pred = y_pred.mean((-3, -2, -1))
        # print(y_pred)
