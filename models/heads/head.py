import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from mmengine import MMLogger

from models.utils.common import ChannelAttention, SpatialAttention, ChannelSelfAttention, SpatialSelfAttention

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
        self.atte = nn.Sequential(
            ChannelAttention(in_channels,reduction_ratio=8),
            # SpatialAttention(),
            # SpatialSelfAttention(dim=384)
        )
        self.dimension_reduction = nn.Linear(14*14*384, 128)
        self.feature_aggregation = nn.GRU(128, 32)

        self.dropout_ratio = dropout_ratio

        self.fc = nn.Sequential(
            nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio > 0 else nn.Identity(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x[0][0]
        logger.debug("head层输入维度: {}".format(x.shape))
        x = self.atte(x)
        x = rearrange(x,'b c t h w -> b t (c h w)')
        x = self.dimension_reduction(x)
        x,_ = self.feature_aggregation(x)
        qlt_score = self.fc(x)
        qlt_score = torch.mean(qlt_score, dim=1)

        return qlt_score
