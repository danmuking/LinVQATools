from torch import nn

from models import logger


class FcHead(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
            self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_hid = nn.Linear(self.in_channels, self.hidden_channels, )
        self.fc_last = nn.Linear(self.hidden_channels, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = x[0][0]
        logger.debug("head层输入维度: {}".format(x.shape))
        x = x.view(x.shape[0], -1)
        logger.debug("head: 特征展开{}".format(x.shape))
        x = self.dropout(x)
        qlt_score = self.fc_hid(x)
        logger.debug("head: Liner{}->{}".format(self.in_channels,self.hidden_channels))
        qlt_score = self.gelu(qlt_score)
        qlt_score = self.fc_last(self.dropout(qlt_score))
        logger.debug("head: Liner{}->{}".format(self.hidden_channels, 1))
        return qlt_score
