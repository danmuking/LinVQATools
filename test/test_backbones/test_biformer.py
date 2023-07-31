from unittest import TestCase

import torch
from einops import rearrange

from models.backbones.biformer.biformer import BiFormer, biformer_tiny, BiFormer3D
from models.backbones.biformer.ops.bra_legacy import TopkRouting, BiLevelRoutingAttention3D


class TestBiFormer(TestCase):
    def test_biformer(self):
        model = BiFormer3D(
        depth=[2, 2, 8, 2],
        embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
        # ------------------------------
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,
        num_classes=1)
        x = torch.zeros(2, 3, 32, 224, 224)
        print(model(x).shape)

    def test_torch(self):
        model = BiLevelRoutingAttention3D(dim=96)
        x = torch.zeros(3,16,56,56,96)
        out = model(x)
        print(out.shape)
