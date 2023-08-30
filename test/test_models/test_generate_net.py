import os
from unittest import TestCase

import torch

from models.generate_net import GenerateNet


class TestGenerateNet(TestCase):
    def test(self):
        os.chdir('../../')
        model = GenerateNet(
            base_x_size=(16, 224, 224),
            vqa_head=dict(name='VQAHead', in_channels=768 * 2, drop_rate=0.5, fc_in=8 * 7 * 7),
            # load_path="/data/ly/code/LinVQATools/pretrained_weights/swin_tiny_patch244_window877_kinetics400_1k.pth"
        )
        video = torch.randn((4, 3, 16, 224, 224))
        gt_video = video
        scores = model(inputs=video, mode="loss", gt_label=torch.randn((4, 1)), gt_video=gt_video[:, :, ::2, ...])
        print(scores)
