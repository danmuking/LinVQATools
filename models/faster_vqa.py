from typing import Optional, Union, Dict

import torch
from mmengine.model import BaseModel

from models.evaluators import DiViDeAddEvaluator
from mmengine import MMLogger, MODELS


@MODELS.register_module()
class FasterVQA(BaseModel):
    def __init__(
            self,
            load_path=None,
            backbone_size="divided",
            backbone_preserve_keys='fragments,resize',
            multi=False,
            layer=-1,
            backbone=dict(resize={"window_size": (4, 4, 4)}, fragments={"window_size": (4, 4, 4)}),
            divide_head=False,
            vqa_head=dict(in_channels=768),
    ):
        super().__init__()
        self.model = DiViDeAddEvaluator(
            backbone=backbone, backbone_size=backbone_size,
            backbone_preserve_keys=backbone_preserve_keys, divide_head=divide_head,
            vqa_head=vqa_head, multi=multi, layer=layer)
        self.logger = MMLogger.get_instance('mmengine', log_level='INFO')
        # 加载预训练权重
        if load_path is not None:
            self._load_weight(load_path)

    def forward(self, inputs: torch.Tensor, data_samples: Optional[list] = None, mode: str = 'tensor') -> Union[
        Dict[str, torch.Tensor], list]:
        if mode == 'loss':
            scores = self.model(inputs)
        return scores

    def _load_weight(self, load_path):
        # 加载预训练参数
        state_dict = torch.load(load_path, map_location='cpu')

        if "state_dict" in state_dict:
            ### migrate training weights from mmaction
            state_dict = state_dict["state_dict"]
            from collections import OrderedDict

            i_state_dict = OrderedDict()
            for key in state_dict.keys():
                if "head" in key:
                    continue
                if "cls" in key:
                    tkey = key.replace("cls", "vqa")
                elif "backbone" in key:
                    i_state_dict[key] = state_dict[key]
                    i_state_dict["fragments_" + key] = state_dict[key]
                    i_state_dict["resize_" + key] = state_dict[key]
                else:
                    i_state_dict[key] = state_dict[key]
            t_state_dict = self.model.state_dict()
            for key, value in t_state_dict.items():
                if key in i_state_dict and i_state_dict[key].shape != value.shape:
                    i_state_dict.pop(key)
            self.logger.info(self.model.load_state_dict(i_state_dict, strict=False))
