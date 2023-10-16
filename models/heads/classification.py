from functools import partial

from einops.layers.torch import Rearrange
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(
            self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        layer_norm = partial(nn.LayerNorm, eps=1e-6)
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, (1, 1, 1)),
            Rearrange('b c t h w -> b t h w c'),
            layer_norm(in_channels),
            Rearrange('b t h w c -> b c t h w'),
            nn.GELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, (1, 2, 2), stride=(1, 2, 2)),
            Rearrange('b c t h w -> b t h w c'),
            layer_norm(in_channels),
            Rearrange('b t h w c -> b c t h w'),
            nn.GELU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, (1, 1, 1)),
            Rearrange('b c t h w -> b t h w c'),
            layer_norm(hidden_channels),
            Rearrange('b t h w c -> b c t h w'),
            nn.GELU(),
            Rearrange('b c t h w -> (b h w) (c t)'),
            nn.Linear(hidden_channels * 8, 49)
        )

    def forward(self, x):
        # NCTHW
        x = x[0][0]

        x = self.dropout(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
