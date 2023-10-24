from functools import partial
from typing import Optional, Union, Dict

import torch
from mmengine import MODELS
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from torch import nn

from models.backbones.video_mae_v2 import VisionTransformer


@MODELS.register_module()
class ClsModel(BaseModel):
    def __init__(
            self,
            load_path=None,
            backbone='faster_vqa',
            **kwargs
    ):
        super().__init__()
        self.model = VisionTransformer(
                patch_size=16,
                embed_dim=384,
                depth=12,
                num_heads=6,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                load_path=load_path,
                num_classes=5,
                use_mean_pooling=True
            )

    def forward(self, inputs: torch.Tensor, data_samples: Optional[list] = None, mode: str = 'tensor', **kargs) -> \
            Union[
                Dict[str, torch.Tensor], list]:
        y = kargs['gt_label'].float().unsqueeze(-1)
        y_c = kargs['y_c'].long().reshape(-1)
        y_r = kargs['y_r'].long().reshape(-1)
        grade = kargs['grade'].long().reshape(-1)
        # print(y.shape)
        if mode == 'loss':
            scores = self.model(inputs, inference=False,
                                reduce_scores=False)
            # y_pred = scores[0]
            # criterion = nn.MSELoss()
            # mse_loss = criterion(y_pred, y)
            # p_loss, r_loss = plcc_loss(y_pred, y), rank_loss(y_pred, y)

            cel = nn.CrossEntropyLoss()

            loss = cel(scores, grade)
            return {'loss': loss,}
        elif mode == 'predict':
            scores = self.model(inputs, inference=True,
                                reduce_scores=False)
            # y_pred = scores[0]
            return (scores,grade)

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore

        losses = {'loss': losses['loss']}
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

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
                    # i_state_dict[key] = state_dict[key]
                    i_state_dict["fragments_" + key] = state_dict[key]
                    # i_state_dict["resize_" + key] = state_dict[key]
                else:
                    i_state_dict[key] = state_dict[key]
            t_state_dict = self.model.state_dict()
            for key, value in t_state_dict.items():
                if key in i_state_dict and i_state_dict[key].shape != value.shape:
                    i_state_dict.pop(key)
            info = self.model.load_state_dict(i_state_dict, strict=False)
            # logger.info(info)
            # self.logger.info(self.model.load_state_dict(i_state_dict, strict=False))
