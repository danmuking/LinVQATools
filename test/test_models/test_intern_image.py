from unittest import TestCase

import torch

from models.intern_image import InternImage


class TestInternImage(TestCase):
    def test(self):
        model = InternImage(
            load_path='/home/ly/code/LinVQATools/pretrained_weights/internimage_t_1k_224.pth',
            depths=[4, 4, 18, 4],
            groups=[4, 8, 16, 32],
        ).cuda()
        video = torch.zeros((2, 3, 16, 224, 224)).cuda()
        scores = model(inputs=video, mode="predict", gt_label=torch.tensor(1))
        print(scores)
