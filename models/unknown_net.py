from typing import Optional, Union, Dict

import torch
from mmengine.model import BaseModel
from torch import nn

from models.backbones.conv_backbone import ConvNeXtV23D
from models.backbones.swin_backbone import SwinTransformer3D


class FusionBlock(nn.Module):
    pass

class UnKnownNet(BaseModel):
    def __init__(self):
        super().__init__()
        self.net1 = ConvNeXtV23D(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        self.net2 = SwinTransformer3D()
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        feature1 = self.net1(inputs)[1:]
        feature2 = self.net2(inputs)[1:-1]
        print([f.shape for f in feature1])
        print([f.shape for f in feature2])
