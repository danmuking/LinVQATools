import torch
from einops import rearrange
from torch import nn

from models import logger


class ClsHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder_to_cls = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(dim, 5)

    def forward(self, x):
        x = x[0][0]
        x = self.encoder_to_cls(x)
        x = self.norm(torch.mean(x, dim=1))
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.softmax(x, dim=-1)
        return x


class VQAHead(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
            self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, fc_in=1568, **kwargs
    ):
        super().__init__()
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
        self.cls_fc = nn.Linear(fc_in, 5)
    def forward(self, x):
        x = x[0][0]
        x = rearrange(x, "b (t h w) c -> b c t h w", t=8, h=14, w=14)
        logger.debug("head层输入维度: {}".format(x.shape))
        qlt_score = self.fc_hid(x)
        logger.debug('head: channel {}->{}'.format(x.shape[1], qlt_score.shape[1]))
        channel_in = qlt_score.shape[1]
        qlt_score = self.fc_last(qlt_score)
        channel_out = qlt_score.shape[1]
        logger.debug('head: channel {}->{}'.format(channel_in, channel_out))

        channel_in = qlt_score.shape
        qlt_score = qlt_score.reshape(qlt_score.shape[0], -1)
        channel_out = qlt_score.shape
        logger.debug('head: 展开 {}->{}'.format(channel_in, channel_out))
        cls_x = qlt_score

        channel_in = qlt_score.shape[1]
        qlt_score = self.fc(qlt_score)
        channel_out = qlt_score.shape[1]
        logger.debug('head: Liner {}->{}'.format(channel_in, channel_out))

        cls_score = self.cls_fc(cls_x)

        return qlt_score,cls_score


class ClsMseHead(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
            self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, fc_in=1568, **kwargs
    ):
        super().__init__()
        # self.cls_head = ClsHead(in_channels)
        self.vqa_head = VQAHead(in_channels, hidden_channels, dropout_ratio, fc_in)

    def forward(self, x):
        # cls_score = self.cls_head(x)
        vqa_score,cls_score = self.vqa_head(x)

        return cls_score, vqa_score
