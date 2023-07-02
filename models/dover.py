from mmengine import MMLogger, MODELS
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
import torch
from functools import partial, reduce
from typing import Optional, Union, Dict
from torch import nn
from global_class.train_recorder import TrainResultRecorder
from models.backbone.conv_backbone import convnext_3d_small, convnext_3d_tiny, convnextv2_3d_pico, convnextv2_3d_femto
from models.faster_vqa import plcc_loss, rank_loss
from models.heads.head import VQAHead
from models.backbone.swin_backbone import SwinTransformer3D as VideoBackbone
from models.backbone.swin_backbone import swin_3d_small, swin_3d_tiny

logger = MMLogger.get_instance('mmengine', log_level='INFO')


class DOVER(nn.Module):
    def __init__(
            self,
            backbone_size="divided",
            backbone_preserve_keys="fragments,resize",
            multi=False,
            layer=-1,
            backbone=dict(
                resize={"window_size": (4, 4, 4)}, fragments={"window_size": (4, 4, 4)}
            ),
            divide_head=False,
            vqa_head=dict(in_channels=768),
            var=False,
    ):
        self.backbone_preserve_keys = backbone_preserve_keys.split(",")
        self.multi = multi
        self.layer = layer
        super().__init__()
        self.technical_backbone = VideoBackbone()
        self.aesthetic_backbone = convnext_3d_tiny(pretrained=True)
        self.vqa_head = VQAHead(**vqa_head)

    def forward(
            self,
            vclips,
            inference=True,
            return_pooled_feats=False,
            reduce_scores=False,
            pooled=False,
            **kwargs
    ):
        if inference:
            self.eval()
            with torch.no_grad():
                scores = []
                feats = {}
                for key in vclips:
                    feat = getattr(self, key.split("_")[0] + "_backbone")(
                        vclips[key], multi=self.multi, layer=self.layer, **kwargs
                    )
                    scores += [feat]
                scores = torch.cat((scores[0], scores[1]), 1)
                scores = self.vqa_head(scores)
            self.train()
            return scores
        else:
            self.train()
            scores = []
            feats = {}
            for key in vclips:
                feat = getattr(self, key.split("_")[0] + "_backbone")(
                    vclips[key], multi=self.multi, layer=self.layer, **kwargs
                )
                scores += [feat]
            scores = torch.cat((scores[0], scores[1]), 1)
            scores = self.vqa_head(scores)
            return scores


@MODELS.register_module()
class DoverWrapper(BaseModel):
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
        args = {'backbone': {'technical': {'type': 'swin_tiny_grpb', 'checkpoint': True, 'pretrained': None},
                             'aesthetic': {'type': 'conv_tiny'}}, 'backbone_preserve_keys': 'technical,aesthetic',
                'divide_head': True, 'vqa_head': {'in_channels': 768, 'hidden_channels': 64}}
        self.model = DOVER(**args)
        load_path = '/home/ly/code/LinVQATools/pretrained_weights/swin_tiny_patch244_window877_kinetics400_1k.pth'
        # 加载预训练权重
        if load_path is not None:
            self._load_weight(load_path)

    def forward(self, inputs: torch.Tensor, data_samples: Optional[list] = None, mode: str = 'tensor', **kargs) -> \
            Union[
                Dict[str, torch.Tensor], list]:
        y = kargs['gt_label'].float().unsqueeze(-1)
        if mode == 'loss':
            y_pred = self.model(inputs, inference=False,
                                reduce_scores=False)
            loss = 0
            p_loss = plcc_loss(y_pred, y)
            r_loss = rank_loss(y_pred, y)
            loss += (
                    p_loss + 0.3 * r_loss
            )

            return {'loss': loss, 'p_loss': p_loss, 'r_loss': r_loss, 'result': [y_pred, y]}
        elif mode == 'predict':
            y_pred = self.model(inputs, inference=True,
                                reduce_scores=False)
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

        losses = {'loss': losses['loss'], 'p_loss': losses['p_loss'], 'r_loss': losses['r_loss']}
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def _load_weight(self, load_path):
        logger.info("加载{}权重".format(load_path))
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
                elif "backbone" in key:
                    i_state_dict[key[9:]] = state_dict[key]
                else:
                    i_state_dict[key] = state_dict[key]
            t_state_dict = self.model.state_dict()
            for key, value in t_state_dict.items():
                if key in i_state_dict and i_state_dict[key].shape != value.shape:
                    i_state_dict.pop(key)
            missing_weight = self.model.technical_backbone.load_state_dict(i_state_dict, strict=False)
            logger.info("missing_weight:{}".format(missing_weight))
