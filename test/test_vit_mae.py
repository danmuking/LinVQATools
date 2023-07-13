from unittest import TestCase

import torch

from models.backbones.vit_mae import VisionTransformer


class TestVisionTransformer(TestCase):
    def test_vit_mae(self):
        model = VisionTransformer(img_size=224,
        patch_size=16,
        embed_dims=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=16,
        norm_cfg=dict(type='LN', eps=1e-6))
        model.init_weights()
        inputs = torch.rand(1, 3, 16, 224, 224)
        ans = model(inputs)
        print(ans)
        print(ans[0].shape)
