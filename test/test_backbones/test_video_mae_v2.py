from unittest import TestCase

import torch

from models.backbones.video_mae_v2 import VisionTransformer


class TestVisionTransformer(TestCase):
    def test(self):
        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            mlp_ratio=4,
            qkv_bias=True,
            num_frames=16,
            norm_cfg=dict(type='LN', eps=1e-6),
            embed_dims=384, depth=12, num_heads=6,
            return_feat_map=True)
        x = torch.zeros((2, 3, 16, 224, 224))
        y = model(x)
        print(y.shape)
    def test_load(self):
        path = '/data/ly/code/LinVQATools/pretrained_weights/vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-25c748fd.pth'
        weight = torch.load(path)
        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            mlp_ratio=4,
            qkv_bias=True,
            num_frames=16,
            norm_cfg=dict(type='LN', eps=1e-6),
            embed_dims=384, depth=12, num_heads=6,
            return_feat_map=True,
            load_path=path
        )
