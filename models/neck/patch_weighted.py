import torch
from einops import rearrange
from torch import nn
from torch.nn import Conv3d


class PatchWeighted(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.batch1 = nn.ModuleList([
            self.dropout,
            Conv3d(768, 128, kernel_size=1),
            # nn.LayerNorm(128),
            nn.GELU(),
            self.dropout,
            Conv3d(128, 1, kernel_size=1),
            nn.GELU(),
        ])
        self.batch2 = nn.ModuleList([
            self.dropout,
            Conv3d(768, 128, kernel_size=1),
            # nn.LayerNorm(128),
            nn.GELU(),
            self.dropout,
            Conv3d(128, 1, kernel_size=1),
            nn.GELU(),
        ])

    def forward(self, x):
        x = x[0][0]
        # print(x.shape)
        score = x
        for m in self.batch1:
            if isinstance(m, nn.LayerNorm):
                b, c, t, h, w = score.shape
                score = rearrange(score, "b c t h w->b (t h w) c")
                score = m(score)
                score = rearrange(score, "b (t h w) c->b c t h w", t=t, h=h, w=w)
            else:
                score = m(score)
        weight = x
        for m in self.batch2:
            if isinstance(m, nn.LayerNorm):
                b, c, t, h, w = weight.shape
                weight = rearrange(weight, "b c t h w->b (t h w) c")
                weight = m(weight)
                weight = rearrange(weight, "b (t h w) c->b c t h w", t=t, h=h, w=w)
            else:
                weight = m(weight)
        score = torch.mul(score, weight)
        return [[score]]
