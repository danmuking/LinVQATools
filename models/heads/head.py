import torch.nn as nn


class VQAHead(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
        pre_pool: whether pre-pool the features or not (True for Aesthetic Attributes, False for Technical Attributes)
    """

    def __init__(
            self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, pre_pool=False, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.pre_pool = pre_pool
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.batch_norm = nn.BatchNorm3d(1536)
        self.fc_hid1 = nn.Conv3d(1536, 768, (1, 1, 1))
        self.fc_hid2 = nn.Conv3d(768, 64, (1, 1, 1))
        self.fc_last = nn.Conv3d(64, 1, (1, 1, 1))
        self.gelu = nn.GELU()
        self.fc = nn.Linear(784, 1)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.gelu(self.fc_hid1(x))
        x = self.dropout(x)
        x = self.gelu(self.fc_hid2(x))
        # x = self.dropout(x)
        qlt_score = self.fc_last(self.dropout(x))
        qlt_score = qlt_score.view(qlt_score.shape[0],-1)
        qlt_score = self.fc(qlt_score)
        return qlt_score