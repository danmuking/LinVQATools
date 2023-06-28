import torch.nn as nn


class VQAHead(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
            self, in_channels=768, hidden_channels=128, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_hid = nn.Conv3d(self.in_channels, self.hidden_channels, (1, 1, 1))
        self.fc_last = nn.Conv3d(self.hidden_channels, 64, (1, 1, 1))
        self.gelu = nn.GELU()
        self.norm1 = nn.BatchNorm3d(128)
        self.norm2 = nn.BatchNorm3d(64)

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(50176, 1)

    def forward(self, x, rois=None):
        x = self.dropout(x)
        qlt_score = self.fc_hid(x)
        qlt_score = self.norm1(qlt_score)
        qlt_score = self.gelu(qlt_score)
        qlt_score = self.fc_last(self.dropout(qlt_score))
        qlt_score = self.norm2(qlt_score)
        qlt_score = self.gelu(qlt_score).reshape(qlt_score.shape[0], -1)
        qlt_score = self.dropout(qlt_score)
        qlt_score = self.fc(qlt_score)
        return qlt_score
