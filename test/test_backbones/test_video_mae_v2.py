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
        x = torch.zeros((1, 3, 16, 224, 224))
        y = model(x)
        print(y.shape)
