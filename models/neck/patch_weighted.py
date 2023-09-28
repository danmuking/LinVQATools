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
            self.dropout,
            Conv3d(768, 64, kernel_size=1),
            nn.GELU(),
            self.dropout,
            Conv3d(64, 1, kernel_size=1),
            nn.GELU(),
        ])
        self.batch2 = nn.ModuleList([
            self.dropout,
            Conv3d(768, 64, kernel_size=1),
            nn.GELU(),
            self.dropout,
            Conv3d(64, 1, kernel_size=1),
            nn.GELU(),
        ])

        self.batch3 = nn.ModuleList([
            self.dropout,
            Conv3d(768, 64, kernel_size=1),
            nn.GELU(),
            self.dropout,
            Conv3d(64, 1, kernel_size=1),
            nn.GELU(),
        ])

    def forward(self, x):
        x = x[0][0]
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

        time_weight = x
        for m in self.batch3:
            if isinstance(m, nn.LayerNorm):
                b, c, t, h, w = time_weight.shape
                time_weight = rearrange(time_weight, "n c d h w -> n d h w c")
                time_weight = m(time_weight)
                time_weight = rearrange(time_weight, "n d h w c -> n c d h w")
            else:
                time_weight = m(time_weight)
        score = torch.mul(score, weight)
        time_weight = rearrange(time_weight, "n c d h w -> n c d (h w)")
        time_weight = time_weight.mean(-1)
        time_weight = time_weight.unsqueeze(-1).unsqueeze(-1)
        score = torch.mul(time_weight, score)
        return [[score]]
