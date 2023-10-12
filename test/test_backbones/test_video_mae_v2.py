from functools import partial
from unittest import TestCase

import torch
from matplotlib import pyplot as plt
from torch import nn

from models.backbones.video_mae_v2 import VisionTransformer, vit_small_patch16_224, get_sinusoid_encoding_table


class TestVisionTransformer(TestCase):
    def test(self):
        path = '/data/ly/code/LinVQATools/pretrained_weights/vit_s_k710_dl_from_giant.pth'
        model = VisionTransformer(
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
            use_mean_pooling=False,
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


    def test_position_emb(self):
        pos = get_sinusoid_encoding_table(
            8*14*14, 384)
        print(pos.shape)
        # from models.backbones.swin_backbone import global_position_index
        # frags_d = torch.arange(1)
        # frags_h = torch.arange(14)
        # frags_w = torch.arange(14)
        # frags = torch.stack(
        #     torch.meshgrid(frags_d, frags_h, frags_w)
        # ).float()  # 3, Fd, Fh, Fw
        # coords = (
        #     torch.nn.functional.interpolate(frags[None], size=(8, 14, 14))
        #     .long()
        #     .permute(0, 2, 3, 4, 1)
        # )
        # coords = coords.abs().sum(-1)
        # coords = coords-(coords.max()/2)
        # coords = coords / coords.max()
        # coords = coords.reshape(1,-1)
        # # print(coords.shape)
        # # print(coords.max(), coords.min())
        #
        # cax = plt.matshow(((pos))[0])
        # plt.gcf().colorbar(cax)
        # plt.show()