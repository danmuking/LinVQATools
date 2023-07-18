from typing import Optional, Union, Dict

import torch
from mmengine.model import BaseModel
from torch import nn

from models.backbones.conv_backbone import ConvNeXtV23D, LayerNorm
from models.backbones.swin_backbone import SwinTransformer3D


class FusionBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Sequential(
            LayerNorm(in_dim, eps=1e-6, data_format="channels_first"),
            nn.Conv3d(in_dim, out_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        self.layer2 = nn.Sequential(
            LayerNorm(in_dim, eps=1e-6, data_format="channels_first"),
            nn.Conv3d(in_dim, out_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

    def forward(self, r1, r2, r3=None):
        x = torch.cat((r1, r2), dim=2)
        if r3 is not None:
            x = torch.cat((r3, x), dim=2)
            res = self.layer2(x)
        x = self.layer1(x)
        if r3 is not None:
            x = x + res
        return x


class UnKnownBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = SwinTransformer3D()
        self.net2 = ConvNeXtV23D(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        self.f_block1 = FusionBlock(192, 384)
        self.f_block2 = FusionBlock(384, 768)
        self.f_block3 = FusionBlock(768, 768)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        x1 = inputs[0]
        x2 = inputs[1]
        feature1 = self.net1(x1)[1:-1]
        feature2 = self.net2(x2)[1:]
        x = self.f_block1(feature1[0], feature2[0])
        x = self.f_block2(feature1[1], feature2[1], x)
        x = self.f_block3(feature1[2], feature2[2], x)
        return x
