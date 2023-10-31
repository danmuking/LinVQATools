import torch
import torch.nn as nn
from einops import rearrange
from mmengine import MMLogger

from models.utils.common import ChannelAttention

logger = MMLogger.get_instance('model', log_level='DEBUG')


class VQAHead(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
            self, in_channels=768, hidden_channels=64, dropout_ratio=0.5,fc_in=1568, **kwargs
    ):
        super().__init__()
        # self.atte = nn.Sequential(
        #     ChannelAttention(in_channels,reduction_ratio=8),
        #     # SpatialAttention(),
        #     # SpatialSelfAttention(dim=384)
        # )
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.fc_hid = nn.Sequential(
            nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio > 0 else nn.Identity(),
            nn.Conv3d(self.in_channels, self.hidden_channels, (1, 1, 1)),
            nn.GELU()
        )
        self.fc_last = nn.Sequential(
            nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio > 0 else nn.Identity(),
            nn.Conv3d(self.hidden_channels, 1, (1, 1, 1)),
            nn.GELU()
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio > 0 else nn.Identity(),
            nn.Linear(fc_in, 1)
        )

    def forward(self, x):
        x = x[0][0]
        logger.debug("head层输入维度: {}".format(x.shape))
        # x = self.atte(x)
        x = rearrange(x, 'b (t h w) c -> b c t h w', t=8, h=14, w=14)
        qlt_score = self.fc_hid(x)
        logger.debug('head: channel {}->{}'.format(x.shape[1],qlt_score.shape[1]))
        channel_in = qlt_score.shape[1]
        qlt_score = self.fc_last(qlt_score)
        channel_out = qlt_score.shape[1]
        logger.debug('head: channel {}->{}'.format(channel_in,channel_out))

        channel_in = qlt_score.shape
        qlt_score = qlt_score.reshape(qlt_score.shape[0], -1)
        channel_out = qlt_score.shape
        logger.debug('head: 展开 {}->{}'.format(channel_in, channel_out))

        channel_in = qlt_score.shape[1]
        qlt_score = self.fc(qlt_score)
        channel_out = qlt_score.shape[1]
        logger.debug('head: Liner {}->{}'.format(channel_in, channel_out))

        return qlt_score

# class VQAHead(nn.Module):
#     def __init__(
#             self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, fc_in=1568, **kwargs
#     ):
#         super().__init__()
#         self.dropout_ratio = dropout_ratio
#         self.in_channels = in_channels
#         self.hidden_channels = hidden_channels
#         self.mlp = nn.ModuleList(
#             [nn.Sequential(
#                 nn.Linear(in_channels,in_channels),
#                 nn.LayerNorm(in_channels,eps=1e-6),
#                 nn.GELU(),
#             ) for x in range(4)]
#         )
#         self.dropout = nn.Dropout(p=self.dropout_ratio)
#         self.fc = nn.Linear(in_channels*4,1)
#
#     def forward(self,x):
#         x = x[0][0]
#         feat = [self.mlp[i](x[i]) for i in range(4)]
#         feat = torch.cat(feat,dim=1).flatten(1)
#         feat = self.dropout(feat)
#         x = self.fc(feat)
#         return x
