import torch
from einops import rearrange, repeat
from torch import nn

from models.backbones.vit_videomae import PretrainVisionTransformerDecoder, Attention, Mlp
from models.utils.common import ChannelAttention
import torch.nn.functional as F
from timm.models.layers import DropPath


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class VQAMlpHead(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
            self, in_channels=384, hidden_channels=64, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()

        self.gap_layer = Block(dim=384, num_heads=8, init_values=0.0)
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.fc_hid = nn.Sequential(
            nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio > 0 else nn.Identity(),
            nn.Linear(self.in_channels, self.hidden_channels),
            nn.GELU()
        )
        self.fc_last = nn.Sequential(
            nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio > 0 else nn.Identity(),
            nn.Linear(self.hidden_channels, 1),
        )

    def forward(self, x):
        x = self.gap_layer(x)
        qlt_score = self.fc_hid(x)
        qlt_score = self.fc_last(qlt_score)
        qlt_score = torch.mean(qlt_score.flatten(1), dim=-1, keepdim=True)

        return qlt_score


def global_std_pool1d(x):
    """2D global standard variation pooling"""
    # x: (B, N, C)
    return torch.std(x, dim=1)


class VQAPoolMlpHead(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
            self, in_channels=512, hidden_channels=64, dropout_ratio=0.5, fc_in=1176, **kwargs
    ):
        super().__init__()

        self.norm = nn.LayerNorm(384, eps=1e-6)
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.encode_to_vqa = nn.Sequential(
            nn.Linear(4608, 2 * 4608),
            nn.Linear(2 * 4608, 4608),
        )
        self.fc_hid = nn.Sequential(
            nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio > 0 else nn.Identity(),
            nn.Linear(4608, 4608 // 4),
            nn.LayerNorm(4608 // 4, eps=1e-6)
        )
        self.fc_last = nn.Sequential(
            nn.Linear(4608 // 4, 1),
        )

    def forward(self, x):
        for i in range(len(x)):
            x[i] = self.norm(torch.cat([torch.mean(x[i], dim=1)], dim=-1))
        x = torch.cat(x, dim=-1)
        qlt_score = self.fc_hid(x)
        qlt_score = self.fc_last(qlt_score)

        return qlt_score
