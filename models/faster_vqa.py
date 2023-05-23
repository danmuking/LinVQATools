from functools import reduce

import torch
from typing import Optional, Union, Dict

from mmengine.model import BaseModel

from models.evaluators import DiViDeAddEvaluator
from mmengine import MMLogger, MODELS


def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
            torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()


def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()


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
            self.logger.info("加载{}权重".format(load_path))
            self._load_weight(load_path)

    def forward(self, inputs: torch.Tensor, data_samples: Optional[list] = None, mode: str = 'tensor', **kargs) -> \
    Union[
        Dict[str, torch.Tensor], list]:
        y = kargs['gt_label'].float().unsqueeze(-1)
        # print(y.shape)
        if mode == 'loss':
            scores = self.model(inputs, inference=False,
                                reduce_scores=False)
            if len(scores) > 1:
                y_pred = reduce(lambda x, y: x + y, scores)
            else:
                y_pred = scores[0]
            y_pred = y_pred.mean((-3, -2, -1))
            # print(y_pred.shape)
            p_loss, r_loss = plcc_loss(y_pred, y), rank_loss(y_pred, y)

            loss = p_loss + 0.3 * r_loss
            return {'loss': loss,'p_loss': p_loss,'r_loss': r_loss}
        elif mode == 'predict':
            scores = self.model(inputs, inference=True,
                                reduce_scores=False)
            if len(scores) > 1:
                y_pred = reduce(lambda x, y: x + y, scores)
            else:
                y_pred = scores[0]
            y_pred = y_pred.mean((-3, -2, -1))
            return y_pred, y

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
