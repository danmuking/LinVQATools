import torch
from einops import rearrange
from torch import nn
from torch.nn import Flatten
import torch.nn.functional as F

from models.backbones.base_swin_backbone import PatchMerging
from models.backbones.video_mae_v2 import Block


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



class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 2, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias,
                                  requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]  # make torchscript happy (cannot use tensor as tuple)
        v = x
        v = v.unsqueeze(1)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # print(attn.shape)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ChannelSelfAttention(nn.Module):
    def __init__(self, dim=1568):
        super(ChannelSelfAttention, self).__init__()
        self.atte = Attention(dim, num_heads=1)

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

        self.conv1 = nn.Conv3d(2, 1, (1, kernel_size, kernel_size), stride=(1, 2, 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv1(scale)
        scale = self.sigmoid(scale)
        x = rearrange(x, 'b c (t n3) (h n1) (w n2) -> b c t h w (n3 n1 n2)', n1=7, n2=7, n3=4)
        scale = rearrange(scale, 'b c (t n3) (h n1) (w n2) -> b c t h w (n3 n1 n2)', n1=7, n2=7, n3=4)
        x = scale * x
        x = rearrange(x, 'b c t h w (n3 n1 n2) -> b c (t n3) (h n1) (w n2)', n1=7, n2=7, n3=4)
        return x


class SpatialSelfAttention(nn.Module):
    def __init__(self, dim=384):
        super(SpatialSelfAttention, self).__init__()

        self.reduce = PatchMerging(embed_dims=dim)
        self.block = Block(dim=dim * 2, num_heads=12, init_values=0., )
        self.norm = nn.LayerNorm(dim * 2)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = rearrange(x, "b c d h w -> b d h w c")
        x = self.reduce(x)
        x = rearrange(x, "b d h w c -> b (d h w) c")
        x = self.block(x)
        x = self.norm(x)
        x = rearrange(x, "b (d h w) c -> b c d h w", d=D, h=7, w=7)
        return x
