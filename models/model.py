from functools import partial
from typing import Union, Dict, Optional

import timm
import torch
from einops import rearrange
from mmengine import MODELS
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from torch import nn

from models.backbones.video_mae_v2 import PreTrainVisionTransformer


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


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        dim = 384
        self.dropout_ratio = 0
        self.img2decoder = nn.Linear(192, 384)
        self.video2decoder = nn.Linear(384, 384)
        self.norm = nn.LayerNorm(384, eps=1e-6)
        self.fusion2decoder = nn.Linear(384, 384)
        self.fc_hid = nn.Sequential(
            nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio > 0 else nn.Identity(),
            nn.Linear(dim, dim // 4),
            nn.GELU()
        )
        self.fc_last = nn.Sequential(
            nn.Linear(dim // 4, 1),
        )

    def forward(self, img_feats, video_feats):
        img_feat = img_feats[-1]
        img_feat = rearrange(img_feat, 'b c h w -> b (h w) c')
        video_feats = video_feats[-1]
        img_feat = self.img2decoder(img_feat)
        video_feats = self.video2decoder(video_feats)
        # 拼接
        x = torch.cat([img_feat, video_feats], dim=1)
        x = self.norm(x)
        x = self.fusion2decoder(x)
        # mean
        x = x.mean(dim=1)
        x = self.fc_hid(x)
        x = self.fc_last(x)

        return x


class Model(nn.Module):
    def __init__(self,
                 mask_ratio=0.,
                 drop_path_rate=0
                 ):
        super(Model, self).__init__()
        self.backbone_embed_dim = 384
        self.vit_backbone = PreTrainVisionTransformer(
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
            use_mean_pooling=False,
            drop_path_rate=drop_path_rate,
            load_path='/data/ly/code/LinVQATools/work_dir/video_mae_vqa/11240027 vit random_cell_mask_75 mae last1/best_SROCC_epoch_358.pth'
        )
        self.cnn_backbone = timm.create_model('tf_efficientnetv2_b0', pretrained=True, features_only=True, )
        self.patch_size = 16
        self.tubelet_size = 2
        self.mask_stride = [1, 1, 1]
        self.input_size = [16, 224]
        # 8 14 14
        self.patches_shape = [self.input_size[0] // self.tubelet_size, self.input_size[1] // self.patch_size,
                              self.input_size[1] // self.patch_size]
        # 8 14 14
        self.mask_shape = [(self.patches_shape[0] // self.mask_stride[0]),
                           (self.patches_shape[1] // self.mask_stride[1]),
                           (self.patches_shape[2] // self.mask_stride[2])]
        self.head = Head()

    def forward(self, inputs, mask):
        video = inputs['video']
        mask = mask.bool()
        B = video.size(0)
        full_mask = mask.reshape(B, *self.mask_shape).repeat_interleave(self.mask_stride[0], dim=1).repeat_interleave(
            self.mask_stride[1], dim=2).repeat_interleave(self.mask_stride[2], dim=3)
        full_mask = full_mask.flatten(2)
        encoder_logits_backbone, feats, patch_embed, x_vis_list = self.vit_backbone(video, ~(full_mask.flatten(1)))
        img = inputs['img']
        img_feat = self.cnn_backbone(img)

        preds_score = self.head(img_feat, feats)
        output = {"preds_score": preds_score}
        return output


class CellRunningMaskAgent(nn.Module):
    def __init__(self, mask_ratio=0):
        super(CellRunningMaskAgent, self).__init__()
        self.patch_num = 8 * 14 * 14
        self.mask_num = int((8 * 14 * 14) * mask_ratio)  # 8*7*7*mark radio
        self.mask_shape = [16 // 2, 14, 14]
        self.mask_stride = [1, 2, 2]
        self.spatial_small_patch_num = (self.mask_shape[1] // self.mask_stride[1]) * (
                self.mask_shape[2] // self.mask_stride[2])  # 8 7 7
        # 8 14 14
        self.test_mask = torch.zeros(self.mask_shape)
        # 8 (2 2) (1*7*7)
        self.test_mask = rearrange(self.test_mask, '(t s0) (h s1) (w s2) -> t (h w) (s0 s1 s2)', s0=self.mask_stride[0],
                                   s1=self.mask_stride[1],
                                   s2=self.mask_stride[2])
        mask_per_patch = self.mask_num // (self.test_mask.size(0) * self.test_mask.size(1))  # 2
        # 每一个patch的mask表
        mask_list = [1 for i in range(mask_per_patch)] + [0 for i in range(self.test_mask.size(2) - mask_per_patch)]
        for t in range(self.test_mask.size(0)):
            offset = t % self.test_mask.size(-1)
            self.test_mask[t, :, :] = torch.Tensor(mask_list[-offset:] + mask_list[:-offset])[None, :]
        self.test_mask = rearrange(self.test_mask, 't (h w) (s0 s1 s2) -> (t s0) (h s1) (w s2)', s0=self.mask_stride[0],
                                   s1=self.mask_stride[1],
                                   s2=self.mask_stride[2],
                                   t=self.mask_shape[0] // self.mask_stride[0],
                                   h=self.mask_shape[1] // self.mask_stride[1],
                                   w=self.mask_shape[2] // self.mask_stride[2], )
        train_mask_list = []
        for i in range(self.mask_stride[1] * self.mask_stride[2]):
            train_mask = torch.zeros(self.mask_shape[0], self.mask_stride[1] * self.mask_stride[2])
            for t in range(train_mask.size(0)):
                offset = (t + i) % train_mask.size(-1)
                train_mask[t, :] = torch.Tensor(mask_list[-offset:] + mask_list[:-offset])
            train_mask_list.append(train_mask)
        self.train_mask = torch.stack(train_mask_list, dim=0)
        self.temporal_shuffle = False
        self.spatial_repeat = True
        self.test_temporal_shuffle = False

    def forward(self, x, mask_shape):
        if isinstance(x, dict):
            x = x['video']
        if self.training:
            if self.spatial_repeat:
                mask_index = torch.randint(self.train_mask.size(0), (x.size(0), 1), device=x.device)
                mask_index = mask_index.repeat(1, self.spatial_small_patch_num).flatten()
            else:
                mask_index = torch.randint(self.train_mask.size(0), (x.size(0), self.spatial_small_patch_num),
                                           device=x.device).flatten()
            selected_mask = self.train_mask.to(x.device)[mask_index, ...].view(x.size(0), self.spatial_small_patch_num,
                                                                               self.train_mask.size(1),
                                                                               self.train_mask.size(2))
            selected_mask = selected_mask.permute(0, 2, 1, 3)
            selected_mask = rearrange(selected_mask, 'b t (h w) (s0 s1 s2) -> b (t s0) (h s1) (w s2)',
                                      s0=self.mask_stride[0],
                                      s1=self.mask_stride[1],
                                      s2=self.mask_stride[2],
                                      t=self.mask_shape[0] // self.mask_stride[0],
                                      h=self.mask_shape[1] // self.mask_stride[1],
                                      w=self.mask_shape[2] // self.mask_stride[2], )
            if self.temporal_shuffle:
                temporal_seed = torch.rand(selected_mask.shape[:2], device=selected_mask.device)
                temporal_index = temporal_seed.argsort(dim=-1)
                selected_mask = torch.gather(selected_mask,
                                             index=temporal_index[:, :, None, None].expand_as(selected_mask), dim=1)
            selected_mask = selected_mask.flatten(1)
            seq_logits = torch.rand(selected_mask.size(0), selected_mask.size(1), device=x.device)
            values, indices = seq_logits.topk(self.mask_num, dim=1, largest=True, sorted=False)
            seq_logits = seq_logits[:, None, :].repeat(1, self.mask_num, 1)
            output = {"seq_logits": seq_logits.detach(), "indices": indices, "mask": 1.0 - selected_mask}
        else:
            selected_mask = self.test_mask.flatten()[None, ...].to(x.device).repeat(x.size(0), 1)
            if self.test_temporal_shuffle:
                selected_mask = selected_mask.view(x.size(0), mask_shape[0], -1)
                temporal_seed = torch.rand(selected_mask.shape[:2], device=selected_mask.device)
                temporal_index = temporal_seed.argsort(dim=-1)
                selected_mask = torch.gather(selected_mask, index=temporal_index[:, :, None].expand_as(selected_mask),
                                             dim=1)
                selected_mask = selected_mask.flatten(1)
            output = {"mask": 1.0 - selected_mask}
        return output


@MODELS.register_module()
class ModelWrapper(BaseModel):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__()
        self.model = Model()
        self.agent = CellRunningMaskAgent(0.75)

    def forward(self, inputs: torch.Tensor, gt_label=None, data_samples: Optional[list] = None, mode: str = 'tensor',
                **kargs) -> \
            Union[
                Dict[str, torch.Tensor], list]:
        B, Clip, C, D, H, W = inputs['video'].shape
        if mode == 'loss':
            y = gt_label.float().unsqueeze(-1)
            video = inputs['video']
            video = rearrange(video, "b clip c t h w -> (b clip) c t h w")
            img = inputs['img']
            img = rearrange(img, "b clip c h w -> (b clip) c h w")
            inputs = {'video': video, 'img': img}
            self.agent.train()
            mask = self.agent(video, [8, 14, 14])['mask']
            mask = mask.reshape(mask.size(0), 8, -1)
            output = self.model(inputs, mask)
            y_pred = output['preds_score']
            criterion = nn.MSELoss()
            mse_loss = criterion(y_pred, y)
            p_loss, r_loss = plcc_loss(y_pred, y), rank_loss(y_pred, y)

            vqa_loss = mse_loss + p_loss + 10 * r_loss
            total_loss = vqa_loss
            return_dict = {'total_loss': total_loss, "vqa_lozz": vqa_loss, 'mse_lozz': mse_loss,
                           'p_lozz': p_loss, 'r_lozz': r_loss}
            return return_dict
        elif mode == 'predict':
            y = gt_label.float().unsqueeze(-1)
            video = inputs['video']
            video = rearrange(video, "b clip c t h w -> (b clip) c t h w")
            img = inputs['img']
            img = rearrange(img, "b clip c h w -> (b clip) c h w")
            inputs = {'video': video, 'img': img}
            self.agent.eval()
            mask = self.agent(video, [8, 14, 14])['mask']
            mask = mask.reshape(mask.size(0), 8, -1)
            output = self.model(inputs, mask)
            y_pred = output['preds_score']
            y_pred = rearrange(y_pred, "(b clip) 1 -> b clip", b=B, clip=Clip)
            y_pred = y_pred.mean(dim=1)
            return y_pred, y
        elif mode == 'tensor':
            inputs = rearrange(inputs, "b clip c t h w -> (b clip) c t h w")
            self.agent.eval()
            mask = self.agent(inputs, [8, 14, 14])['mask']
            mask = mask.reshape(mask.size(0), 8, -1)
            output = self.model(inputs, mask)
            y_pred = output['preds_score']
            y_pred = rearrange(y_pred, "(b clip) 1 -> b clip", b=B, clip=Clip)
            y_pred = y_pred.mean(dim=1)
            return y_pred

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

        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars