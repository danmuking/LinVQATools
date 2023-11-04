import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from mmengine import MMLogger

from models.utils.common import ChannelAttention, SpatialAttention, ChannelSelfAttention, SpatialSelfAttention

logger = MMLogger.get_instance('model', log_level='DEBUG')



def global_std_pool3d(x):
    """3D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], x.size()[2], -1, 1), dim=3, keepdim=True)
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

        self.dimension_reduction = nn.Linear(384*2, 128)
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
        features_mean = nn.functional.adaptive_avg_pool2d(x, 1).flatten(2).permute(0, 2, 1)
        features_std = global_std_pool3d(x).flatten(2).permute(0, 2, 1)
        x = torch.cat([features_mean, features_std], dim=2)
        # x = rearrange(x,'b c t-> b t c')
        x = self.dimension_reduction(x)
        x,_ = self.feature_aggregation(x)
        qlt_score = self.fc(x)
        qlt_score = torch.mean(qlt_score, dim=1)

        return qlt_score
