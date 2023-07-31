import torch
from typing import Optional, Union, Dict

from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from torch import nn

from global_class.train_recorder import TrainResultRecorder
from models.backbones.biformer.biformer import BiFormer3D
from mmengine import MODELS

from models.heads.fc import FcHead



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
class BiFormerArc(BaseModel):
    def __init__(
            self,
            load_path=None,
            vqa_head=dict(in_channels=512),
            **kwargs
    ):
        super().__init__()
        self.backbone = BiFormer3D(
            depth=[2, 2, 8, 2],
            embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
            # ------------------------------
            n_win=7,
            kv_downsample_mode='identity',
            kv_per_wins=[-1, -1, -1, -1],
            topks=[1, 4, 16, -2],
            side_dwconv=5,
            before_attn_dwconv=3,
            layer_scale_init_value=-1,
            qk_dims=[64, 128, 256, 512],
            head_dim=32,
            param_routing=False, diff_routing=False, soft_routing=False,
            pre_norm=True,
            pe=None,
            load_path=load_path
        )
        self.head = FcHead(**vqa_head)
        # # 加载预训练权重
        # if load_path is not None:
        #     self._load_weight(load_path)

    def forward(self, inputs: torch.Tensor, data_samples: Optional[list] = None, mode: str = 'tensor', **kargs) -> \
            Union[
                Dict[str, torch.Tensor], list]:
        y = kargs['gt_label'].float().unsqueeze(-1)
        # print(y.shape)
        if mode == 'loss':
            scores = self.backbone(x=inputs)
            scores = self.head(scores)
            y_pred = scores
            # print(y_pred)
            criterion = nn.MSELoss()
            mse_loss = criterion(y_pred, y)
            p_loss, r_loss = plcc_loss(y_pred, y), rank_loss(y_pred, y)

            loss = mse_loss + p_loss + 3 * r_loss
            return {'loss': loss, 'mse_loss': mse_loss, 'p_loss': p_loss, 'r_loss': r_loss, 'result': [y_pred, y]}
        elif mode == 'predict':
            scores = self.backbone(x=inputs)
            # print(scores.shape)
            scores = self.head(scores)
            y_pred = scores
            return y_pred, y

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
        result = losses['result']
        recorder = TrainResultRecorder.get_instance('mmengine')
        recorder.iter_y_pre = result[0]
        recorder.iter_y = result[1]

        losses = {'loss': losses['loss'], 'mse_loss': losses['mse_loss'], 'p_loss': losses['p_loss'],
                  'r_loss': losses['r_loss']}
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
                    i_state_dict[key] = state_dict[key]
                    i_state_dict["fragments_" + key] = state_dict[key]
                    i_state_dict["resize_" + key] = state_dict[key]
                else:
                    i_state_dict[key] = state_dict[key]
            t_state_dict = self.model.state_dict()
            for key, value in t_state_dict.items():
                if key in i_state_dict and i_state_dict[key].shape != value.shape:
                    i_state_dict.pop(key)
            self.model.load_state_dict(i_state_dict, strict=False)
            # self.logger.info(self.model.load_state_dict(i_state_dict, strict=False))
