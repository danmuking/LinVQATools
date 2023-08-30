from functools import partial
from unittest import TestCase

import torch
from torch import nn

from models.backbones.video_mae_v2 import VisionTransformer, vit_small_patch16_224


class TestVisionTransformer(TestCase):
    def test(self):
        path = '/data/ly/code/LinVQATools/pretrained_weights/vit_s_k710_dl_from_giant.pth'
        model = VisionTransformer(
            patch_size=8,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
            use_mean_pooling=True,
            load_path = path,
        )
        x = torch.zeros((2, 3, 16, 224, 224))
        y = model(x)
        print(y[0][0].shape)
    def test_load(self):
        path = '/data/ly/code/LinVQATools/pretrained_weights/vit_s_k710_dl_from_giant.pth'
        weight = torch.load(path)['module']
        # print(weight.keys())
        model = VisionTransformer(
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            load_path=path,
            num_classes=0,
            use_mean_pooling=False
        )
