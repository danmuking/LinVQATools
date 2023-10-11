import torch
from torch import nn

from models import logger


class MeanHead(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
            self, **kwargs
    ):
        super().__init__()

    def forward(self, x):
        x = x[0][0]
        logger.debug("head层输入维度: {}".format(x.shape))
        x = x.reshape(x.shape[0], -1)
        return torch.mean(x, dim=1, keepdim=True)
