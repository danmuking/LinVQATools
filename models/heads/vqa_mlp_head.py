from torch import nn

from models.utils.common import ChannelAttention


class VQAMlpHead(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
            self, in_channels=512, hidden_channels=64, dropout_ratio=0.5,fc_in=784*2, **kwargs
    ):
        super().__init__()
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
        self.fc = nn.Sequential(
            nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio > 0 else nn.Identity(),
            nn.Linear(fc_in, 1)
        )

    def forward(self, x):
        qlt_score = self.fc_hid(x)
        qlt_score = self.fc_last(qlt_score)
        qlt_score = qlt_score.reshape(qlt_score.shape[0], -1)
        qlt_score = self.fc(qlt_score)

        return qlt_score
