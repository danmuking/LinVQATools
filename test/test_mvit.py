import os
from collections import OrderedDict
from unittest import TestCase

import torch
from mmengine import MODELS
from models.backbones.mvit import MViT


class TestMViT(TestCase):
    def testMViT(self):
        os.chdir('../')
        cfg = dict(type='MViT', arch='tiny'
                   ,pretrained='/home/ly/code/LinVQATools/pretrained_weights/mvit-small-p244_16x4x1_kinetics400-rgb_20221021-9ebaaeed.pth')
        model = MODELS.build(cfg)
        model.init_weights()
        inputs = torch.rand(1, 3, 16, 224, 224)
        outputs = model(inputs)
        # print(outputs)
        # print(outputs[0][0].shape)
        print(model.state_dict().keys())

    def test_process_weight(self):
        """
        处理一下预训练权重
        """
        path = '/home/ly/code/LinVQATools/pretrained_weights/mvit-small-p244_16x4x1_kinetics400-rgb_20221021-9ebaaeed.pth'
        weight = torch.load(path)
        print(weight.keys())
        print(weight['state_dict'].keys())
        weight = weight['state_dict']
        t_dict = OrderedDict()
        for k,v in weight.items():
            if k.startswith('backbone'):
                print(k)

