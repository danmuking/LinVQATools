import torch
from einops import rearrange
from torch import nn
from torch.nn import Conv3d

from models.backbones.swin_backbone import BasicLayer


class PatchWeighted(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.batch1 = nn.ModuleList([
            BasicLayer(
                dim=768,
                depth=2,
                num_heads=24,
                window_size=(8, 7, 7),
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                drop=0.0,
                attn_drop=0.0,
                # drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                drop_path=0,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=True,
                jump_attention=False,
                frag_bias=False,
            ),
            nn.LayerNorm(768),
            self.dropout,
            Conv3d(768, 64, kernel_size=1),
            nn.GELU(),
            self.dropout,
            Conv3d(64, 2, kernel_size=1),
            nn.GELU(),
        ])
        self.batch2 = nn.ModuleList([
            BasicLayer(
                dim=768,
                depth=2,
                num_heads=24,
                window_size=(8, 7, 7),
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                drop=0.0,
                attn_drop=0.0,
                # drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                drop_path=0,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=True,
                jump_attention=False,
                frag_bias=False,
            ),
            nn.LayerNorm(768),
            self.dropout,
            Conv3d(768, 64, kernel_size=1),
            nn.GELU(),
            self.dropout,
            Conv3d(64, 2, kernel_size=1),
            nn.GELU(),
        ])

    def forward(self, x):
        x = x[0][0]
        # print(x.shape)
        score = x
        for m in self.batch1:
            if isinstance(m, nn.LayerNorm):
                b, c, t, h, w = score.shape
                score = rearrange(score, "n c d h w -> n d h w c")
                score = m(score)
                score = rearrange(score, "n d h w c -> n c d h w")
            else:
                score = m(score)
        weight = x
        for m in self.batch2:
            if isinstance(m, nn.LayerNorm):
                b, c, t, h, w = weight.shape
                weight = rearrange(weight, "n c d h w -> n d h w c")
                weight = m(weight)
                weight = rearrange(weight, "n d h w c -> n c d h w")
            else:
                weight = m(weight)
        score = torch.mul(score, weight)
        return [[score]]
