import torch
from typing import Optional, Union, Dict

from einops import rearrange
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from torch import nn

from global_class.train_recorder import TrainResultRecorder
from models.backbones.generate_backbone import SwinTransformer3D
from mmengine import MODELS
from models.generator.networks import UpsamplingGenerator
from models.heads import VQAHead


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
class GenerateNet(BaseModel):
    def __init__(
            self,
            load_path=None,
            base_x_size=(32, 224, 224),
            vqa_head=dict(name='VQAHead', in_channels=768),
            window_size=(8, 7, 7),
            **kwargs
    ):
        super().__init__()
        self.backbone = SwinTransformer3D(
            window_size=window_size,
            base_x_size=base_x_size,
            load_path=load_path
        )
        self.vqa_head = VQAHead(**vqa_head)
        self.generate_head = UpsamplingGenerator(input_nc=768, output_nc=3)

        self.backbone.load(load_path=load_path)

    def forward(self, inputs: torch.Tensor, data_samples: Optional[list] = None, mode: str = 'tensor', **kargs) -> \
            Union[
                Dict[str, torch.Tensor], list]:
        y = kargs['gt_label'].float().unsqueeze(-1)
        if mode == 'loss':
            self.train()
            feat = self.backbone(inputs)

            generate_feat = feat[0][1]
            generate_video = self.generate_head(generate_feat)
            mae_loss = nn.L1Loss()
            gt_video = kargs['gt_video']
            gt_video = rearrange(gt_video, "n c t h w-> n (c t h w)")
            generate_video = rearrange(generate_video, "n c t h w-> n (c t h w)")
            generate_loss = mae_loss(generate_video, gt_video)

            vqa_feat = feat[0][0]
            vqa_feat = [[torch.cat((vqa_feat, generate_feat), dim=1)]]
            vqa_scores = self.vqa_head(vqa_feat)
            y_pred = vqa_scores
            criterion = nn.MSELoss()
            mse_loss = criterion(y_pred, y)
            p_loss, r_loss = plcc_loss(y_pred, y), rank_loss(y_pred, y)
            vqa_loss = mse_loss + p_loss + 3 * r_loss

            loss = vqa_loss + generate_loss

            return {'loss': loss, 'mse_loss': mse_loss, 'p_loss': p_loss,
                    'r_loss': 3 * r_loss, 'vqa_loss': vqa_loss, 'generate_loss': generate_loss,
                    'result': [y_pred, y]}
        elif mode == 'predict':
            self.eval()
            with torch.no_grad():
                feat = self.backbone(inputs)
                generate_feat = feat[0][1]
                vqa_feat = feat[0][0]
                vqa_feat = [[torch.cat((vqa_feat, generate_feat), dim=1)]]
                vqa_scores = self.vqa_head(vqa_feat)
                y_pred = vqa_scores
            self.train()
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
                  'r_loss': losses['r_loss'],'vqa_loss': losses['vqa_loss'],'generate_loss': losses['generate_loss']}
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
