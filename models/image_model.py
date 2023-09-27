import torch
from typing import Optional, Union, Dict

from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from torch import nn

from global_class.train_recorder import TrainResultRecorder
from models.backbones.cpn_net import CPNet
from models.evaluators import DiViDeAddEvaluator
from mmengine import MODELS
from models import logger


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
class ImageModel(BaseModel):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__()
        self.model = CPNet(args=None)

    def forward(self, inputs: torch.Tensor, data_samples: Optional[list] = None, mode: str = 'tensor', **kargs) -> \
            Union[
                Dict[str, torch.Tensor], list]:
        y = kargs['gt_label'].float().unsqueeze(-1)
        name = kargs['name']
        # print(name)
        # gt_motion = kargs['camera_motion'].float().unsqueeze(-1)
        # print(y.shape)
        if mode == 'loss':
            scores = self.model(inputs)
            y_pred = scores
            # motion = scores[1]
            criterion = nn.MSELoss()
            mse_loss = criterion(y_pred, y)
            loss = mse_loss
            return {'loss': loss, 'result': [y_pred, y]}
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
