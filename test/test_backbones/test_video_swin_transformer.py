from unittest import TestCase

import torch

from models.backbones.swin_backbone import SwinTransformer3D


class TestVideoSwinTransformer(TestCase):
    def test(self):
        model = SwinTransformer3D(base_x_size=(16, 224, 224))
        x = torch.zeros((1,3,16,224,224))
        y = model(x)
        print(y)

    def test_load(self):
        path = '/data/ly/code/LinVQATools/pretrained_weights/biformer_tiny_best.pth'
        weight = torch.load(path)
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
        # print(weight['model'].keys())

        t_state_dict = model.state_dict()
        s_state_dict = torch.load(path)["model"]
        from collections import OrderedDict
        for key in t_state_dict.keys():
            if key not in s_state_dict:
                print(key)
                continue
            if t_state_dict[key].shape != s_state_dict[key].shape:
                print(t_state_dict[key].shape, s_state_dict[key].shape)
                t = t_state_dict[key].shape[2]
                s_state_dict[key] = s_state_dict[key].unsqueeze(2).repeat(1, 1, t, 1, 1) / t
        info = model.load_state_dict(s_state_dict, strict=False)
        print(info)