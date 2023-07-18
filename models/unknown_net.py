from typing import Optional, Union, Dict

import torch
from mmengine.model import BaseModel

from models.backbones.unknown_backbone import UnKnownBackbone
from models.heads.head import VQAHead


class UnknownNet(BaseModel):
    def __init__(self):
        super().__init__()
        self.backbone = UnKnownBackbone()
        self.head = VQAHead()

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        if mode == 'predict':
            x = self.backbone(inputs)
            x = self.head(x)
            y_pred = x[0]
            return y_pred
