from typing import Optional, Union, Dict

import torch
from mmengine import MODELS
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from torch import nn

from global_class.train_recorder import TrainResultRecorder
from models.backbones.intern_image.intern_image import InternImage as backbone
from models.faster_vqa import plcc_loss, rank_loss
from models import logger

@MODELS.register_module()
class InternImage(BaseModel):
    def __init__(
            self,
            load_path=None,
            num_classes=1,
            channels=64,
            depths=[4, 4, 18, 4],
            groups=[4, 8, 16, 32],
            offset_scale=1.0,
            mlp_ratio=4.0
    ):
        super().__init__()
        self.model = backbone(
            core_op='DCNv3',
            num_classes=num_classes,
            channels=channels,
            depths=depths,
            groups=groups,
            offset_scale=offset_scale,
            mlp_ratio=mlp_ratio,
            load_path=load_path
        )

    def forward(self, inputs: torch.Tensor, data_samples: Optional[list] = None, mode: str = 'tensor', **kargs) -> \
            Union[
                Dict[str, torch.Tensor], list]:
        y = kargs['gt_label'].float().unsqueeze(-1)
        name = kargs['name']
        # print(y.shape)
        if mode == 'loss':
            scores = self.model(inputs)
            y_pred = scores
            criterion = nn.MSELoss()
            mse_loss = criterion(y_pred, y)
            loss = mse_loss
            return {'loss': loss,'result': [y_pred, y] }
        elif mode == 'predict':
            scores = self.model(inputs)
            y_pred = scores
            return y_pred, y,name

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

        # 略作修改，适配一下train hook
        # result = losses['result']
        # recorder = TrainResultRecorder.get_instance('mmengine')
        # recorder.iter_y_pre = result[0]
        # recorder.iter_y = result[1]

        losses = {'loss': losses['loss']}
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars
