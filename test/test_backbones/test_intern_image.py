import os
from unittest import TestCase

import torch

from models.backbones.intern_image.build import build_model
from models.backbones.intern_image.intern_image import InternImage


class TestInternImage(TestCase):
    def test_intern_image_backbone(self):
        os.chdir('../../')
        config = dict(
            model=dict(
                type='intern_image',
                drop_path_rate=0.1,
                intern_image=dict(
                    core_op='DCNv3',
                    depths=[4, 4, 18, 4],
                    groups=[4, 8, 16, 32],
                    channels=64,
                    offset_scale=1.0,
                    mlp_ratio=4.0
                )
            )
        )
        model = InternImage(
            core_op='DCNv3',
            num_classes=1,
            channels=64,
            depths=[4, 4, 18, 4],
            groups=[4, 8, 16, 32],
            offset_scale=1.0,
            mlp_ratio=4.0,
        ).cuda()
        print(model)
        x = torch.zeros((2, 3, 224, 224)).cuda()
        ans = model(x)
        print(ans[0])

    def test(self):
        s_dict = torch.load('/home/ly/code/LinVQATools/pretrained_weights/internimage_t_1k_224.pth', map_location='cpu')
        model = InternImage(
            core_op='DCNv3',
            num_classes=1,
            channels=64,
            depths=[4, 4, 18, 4],
            groups=[4, 8, 16, 32],
            offset_scale=1.0,
            mlp_ratio=4.0,
        )
        print(s_dict['model'].keys())
        print(model.state_dict().keys())
