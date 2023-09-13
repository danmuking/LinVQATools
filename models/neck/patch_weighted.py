import torch
from torch import nn
from torch.nn import Conv3d


class PatchWeighted(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch1 = nn.Sequential(
            Conv3d(768, 128, kernel_size=1),
            nn.GELU(),
            Conv3d(128, 1, kernel_size=1),
        )
        self.batch2 = nn.Sequential(
            Conv3d(768, 128, kernel_size=1),
            nn.GELU(),
            Conv3d(128, 1, kernel_size=1),
        )

    def forward(self, x):
        x = x[0][0]
        # print(x.shape)
        score = self.batch1(x)
        weight = self.batch2(x)
        return [[torch.mul(score, weight)]]
