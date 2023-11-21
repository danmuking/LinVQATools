import torch
from einops import rearrange, repeat
from torch import nn

from models.backbones.vit_videomae import PretrainVisionTransformerDecoder
from models.utils.common import ChannelAttention


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    From https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py.
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    From https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py.
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, ff_dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Self-attention module.
    Currently supports both full self-attention on all the input tokens,
    or only-spatial/only-temporal self-attention.

    See Anurag Arnab et al.
    ViVIT: A Video Vision Transformer.
    and
    Gedas Bertasius, Heng Wang, Lorenzo Torresani.
    Is Space-Time Attention All You Need for Video Understanding?

    Modified from
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(
            self,
            dim,
            num_heads=12,
            attn_dropout=0.,
            ff_dropout=0.,
            einops_from=None,
            einops_to=None,
            **einops_dims,
    ):
        super().__init__()
        self.num_heads = num_heads
        dim_head = dim // num_heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.ff_dropout = nn.Dropout(ff_dropout)

        if einops_from is not None and einops_to is not None:
            self.partial = True
            self.einops_from = einops_from
            self.einops_to = einops_to
            self.einops_dims = einops_dims
        else:
            self.partial = False

    def forward(self, x):
        if self.partial:
            return self.forward_partial(
                x,
                self.einops_from,
                self.einops_to,
                **self.einops_dims,
            )
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # import os
        # for i in range(12):
        #     if not os.path.exists(f"./debug/transformer_visualization/layer_{i}.pyth"):
        #         break
        # torch.save(attn,f"./debug/transformer_visualization/layer_{i}.pyth")
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x

    def forward_partial(self, x, einops_from, einops_to, **einops_dims):
        h = self.num_heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q *= self.scale

        # splice out classification token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let classification token attend to key / values of all patches across time and space
        cls_attn = (cls_q @ k.transpose(1, 2)).softmax(-1)
        cls_attn = self.attn_dropout(cls_attn)
        cls_out = cls_attn @ v

        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r=r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        # attention
        attn = (q_ @ k_.transpose(1, 2)).softmax(-1)
        attn = self.attn_dropout(attn)
        x = attn @ v_

        # merge back time or space
        x = rearrange(x, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        x = torch.cat((cls_out, x), dim=1)

        # merge back the heads
        x = rearrange(x, '(b h) n d -> b n (h d)', h=h)

        # combine heads out
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x


class BaseTransformerLayer(nn.Module):
    def __init__(self, cfg, dim_override=None, num_heads_override=None, attn_dropout_override=None,
                 ff_dropout_override=None, mlp_mult_override=None, drop_path_rate=0.0):
        """
        Args:
            cfg             (Config): global config object.
            drop_path_rate  (float): rate for drop path.
                See models/base/base_blocks.py L897-928.
        """
        super().__init__()

        dim = dim_override if dim_override is not None else cfg.VIDEO.BACKBONE.NUM_FEATURES
        num_heads = num_heads_override if num_heads_override is not None else cfg.VIDEO.BACKBONE.NUM_HEADS
        attn_dropout = attn_dropout_override if attn_dropout_override is not None else cfg.VIDEO.BACKBONE.ATTN_DROPOUT
        ff_dropout = ff_dropout_override if ff_dropout_override is not None else cfg.VIDEO.BACKBONE.FF_DROPOUT
        mlp_mult = mlp_mult_override if mlp_mult_override is not None else cfg.VIDEO.BACKBONE.MLP_MULT
        drop_path = drop_path_rate

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout
        )
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(dim=dim, mult=mlp_mult, ff_dropout=ff_dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x


class VQAMlpHead(nn.Module):
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

        # self.num_features = 512
        # self.num_heads = 4
        # drop_path = 0
        # depth = 2
        # attn_drop_out = 0
        # ffn_drop_out = 0.0
        # mlp_mult_override = 4
        #
        # dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        # self.mask_inv_test = False
        # self.blocks = nn.ModuleList([
        #     BaseTransformerLayer(cfg=None, num_heads_override=self.num_heads, dim_override=self.num_features,
        #                          drop_path_rate=dpr[i], attn_dropout_override=attn_drop_out,
        #                          ff_dropout_override=ffn_drop_out, mlp_mult_override=mlp_mult_override)
        #     for i in range(depth)])
        # self.norm = nn.LayerNorm(self.num_features, eps=1e-6)

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
            nn.GELU()
        )


    def forward(self, x):
        # out = x
        # for blk in self.blocks:
        #     out = blk(out)
        # out = self.norm(out)
        # x = out

        qlt_score = self.fc_hid(x)
        qlt_score = self.fc_last(qlt_score)
        qlt_score = torch.mean(qlt_score.flatten(1),dim=-1,keepdim=True)

        return qlt_score


def global_std_pool1d(x):
    """2D global standard variation pooling"""
    # x: (B, N, C)
    return torch.std(x,dim=1)


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

        self.norm = nn.LayerNorm(384,eps=1e-6)
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.encode_to_vqa = nn.Sequential(
            nn.Linear(4608,2*4608),
            nn.Linear(2*4608, 4608),
        )
        self.fc_hid = nn.Sequential(
            nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio > 0 else nn.Identity(),
            nn.Linear(4608, 4608//4),
            nn.LayerNorm(4608//4,eps=1e-6)
        )
        self.fc_last = nn.Sequential(
            nn.Linear(4608//4, 1),
        )

    def forward(self, x):
        for i in range(len(x)):
            x[i] = self.norm(torch.cat([torch.mean(x[i],dim=1)],dim=-1))
        x = torch.cat(x,dim=-1)
        qlt_score = self.fc_hid(x)
        qlt_score = self.fc_last(qlt_score)

        return qlt_score
