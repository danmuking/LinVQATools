import torch
from einops import rearrange
from torch import nn
from torch.nn import Flatten
import torch.nn.functional as F

from models.backbones.video_mae_v2 import Attention


class ChannelAttention(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelAttention, self).__init__()
        self.gate_channels = gate_channels
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = self.avg_pool(x)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = self.max_pool(x)
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_raw).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale


class ChannelSelfAttention(nn.Module):
    def __init__(self, dim=1568):
        super(ChannelSelfAttention, self).__init__()
        self.atte = Attention(dim,num_heads=1)

    def forward(self, x):
        """
        :param x: (b,c,t,h,w)
        """
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> b c (t h w)')
        x = self.atte(x)
        x = rearrange(x, 'b c (t h w) -> b c t h w', t=T, h=H, w=W)

        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=2):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(2, 1, (1, kernel_size, kernel_size),stride=(1,2,2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv1(scale)
        scale = self.sigmoid(scale)
        x = rearrange(x, 'b c (t n3) (h n1) (w n2) -> b c t h w (n3 n1 n2)', n1=7,n2=7,n3=4)
        scale = rearrange(scale, 'b c (t n3) (h n1) (w n2) -> b c t h w (n3 n1 n2)', n1=7,n2=7,n3=4)
        x = scale * x
        x = rearrange(x, 'b c t h w (n3 n1 n2) -> b c (t n3) (h n1) (w n2)', n1=7, n2=7, n3=4)
        return x
