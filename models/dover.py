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
        for key, hypers in backbone.items():
            print(backbone_size)
            if key not in self.backbone_preserve_keys:
                continue
            if backbone_size == "divided":
                t_backbone_size = hypers["type"]
            else:
                t_backbone_size = backbone_size
            if t_backbone_size == "swin_tiny":
                b = swin_3d_tiny(**backbone[key])
            elif t_backbone_size == "swin_tiny_grpb":
                # to reproduce fast-vqa
                b = VideoBackbone()
            elif t_backbone_size == "swin_tiny_grpb_m":
                # to reproduce fast-vqa-m
                b = VideoBackbone(window_size=(4, 4, 4), frag_biases=[0, 0, 0, 0])
            elif t_backbone_size == "swin_small":
                b = swin_3d_small(**backbone[key])
            elif t_backbone_size == "conv_tiny":
                b = convnext_3d_tiny(pretrained=True)
            elif t_backbone_size == "conv_small":
                b = convnext_3d_small(pretrained=True)
            elif t_backbone_size == "conv_femto":
                b = convnextv2_3d_femto(pretrained=True)
            elif t_backbone_size == "conv_pico":
                b = convnextv2_3d_pico(pretrained=True)
            elif t_backbone_size == "xclip":
                raise NotImplementedError
                # b = build_x_clip_model(**backbone[key])
            else:
                raise NotImplementedError
            print("Setting backbone:", key + "_backbone")
            setattr(self, key + "_backbone", b)
        if divide_head:
            for key in backbone:
                pre_pool = False  # if key == "technical" else True
                if key not in self.backbone_preserve_keys:
                    continue
                b = VQAHead(pre_pool=pre_pool, **vqa_head)
                print("Setting head:", key + "_head")
                setattr(self, key + "_head", b)
        else:
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
                    if hasattr(self, key.split("_")[0] + "_head"):
                        scores += [getattr(self, key.split("_")[0] + "_head")(feat)]
                    else:
                        scores += [getattr(self, "vqa_head")(feat)]
                    if return_pooled_feats:
                        feats[key] = feat.mean((-3, -2, -1))
                if reduce_scores:
                    if len(scores) > 1:
                        scores = reduce(lambda x, y: x + y, scores)
                    else:
                        scores = scores[0]
                    if pooled:
                        scores = torch.mean(scores, (1, 2, 3, 4))
            self.train()
            if return_pooled_feats:
                return scores, feats
            return scores
        else:
            self.train()
            scores = []
            feats = {}
            for key in vclips:
                feat = getattr(self, key.split("_")[0] + "_backbone")(
                    vclips[key], multi=self.multi, layer=self.layer, **kwargs
                )
                if hasattr(self, key.split("_")[0] + "_head"):
                    scores += [getattr(self, key.split("_")[0] + "_head")(feat)]
                else:
                    scores += [getattr(self, "vqa_head")(feat)]
                if return_pooled_feats:
                    feats[key] = feat.mean((-3, -2, -1))
            if reduce_scores:
                if len(scores) > 1:
                    scores = reduce(lambda x, y: x + y, scores)
                else:
                    scores = scores[0]
                if pooled:
                    print(scores.shape)
                    scores = torch.mean(scores, (1, 2, 3, 4))
                    print(scores.shape)

            if return_pooled_feats:
                return scores, feats
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
            # self.logger.info("加载{}权重".format(load_path))
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
            loss = 0
            p_loss_a = plcc_loss(scores[0].mean((-3, -2, -1)), y)
            p_loss_b = plcc_loss(scores[1].mean((-3, -2, -1)), y)
            r_loss_a = rank_loss(scores[0].mean((-3, -2, -1)), y)
            r_loss_b = rank_loss(scores[1].mean((-3, -2, -1)), y)
            loss += (
                    p_loss_a + p_loss_b + 0.3 * r_loss_a + 0.3 * r_loss_b
            )

            return {'loss': loss, 'p_loss_a': p_loss_a, 'r_loss_a': r_loss_a, 'p_loss_b': p_loss_b,
                    'r_loss_b': r_loss_b, 'result': [y_pred, y]}
        elif mode == 'predict':
            scores = self.model(inputs, inference=True,
                                reduce_scores=True)
            # print(scores)
            # if len(scores) > 1:
            #     y_pred = reduce(lambda x, y: x + y, scores)
            # else:
            #     y_pred = scores[0]
            y_pred = scores.mean((-3, -2, -1))
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

        losses = {'loss': losses['loss'], 'p_loss_a': losses['p_loss_a'], 'r_loss_a': losses['r_loss_a'],
                  'p_loss_b': losses['p_loss_b'],
                  'r_loss_b': losses['r_loss_b']}
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
