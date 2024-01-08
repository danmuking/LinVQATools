import os
from collections import OrderedDict
from functools import partial
from typing import Union, Dict, Optional

import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from mmengine.visualization import Visualizer
from torch import nn

from data.default_dataset import SingleBranchDataset
from models.faster_vqa import plcc_loss, rank_loss
from models.heads.vqa_mlp_head import VQAMlpHead, VQAPoolMlpHead
from models.backbones.vit_videomae import get_sinusoid_encoding_table, PatchEmbed, Block
import torch.utils.checkpoint as cp

res = []


class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1536,
            embed_dim=384,
            depth=4,
            num_heads=8,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0,
            attn_drop_rate=0,
            drop_path_rate=0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=0.0,
            tubelet_size=2,
            use_learnable_pos_emb=False,
    ):
        super().__init__()
        from models.backbones.vit_videomae import Block
        img_size = img_size
        patch_size = patch_size
        in_chans = in_chans
        num_classes = num_classes
        embed_dim = embed_dim
        depth = depth
        num_heads = num_heads
        mlp_ratio = mlp_ratio
        qkv_bias = qkv_bias
        qk_scale = qk_scale
        drop_rate = drop_rate
        attn_drop_rate = attn_drop_rate
        drop_path_rate = drop_path_rate
        norm_layer = norm_layer
        init_values = init_values
        tubelet_size = tubelet_size
        use_learnable_pos_emb = use_learnable_pos_emb

        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x


class PreTrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 head_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_mean_pooling=True,
                 with_cp=False,
                 cos_attn=False,
                 load_path=None):
        super().__init__()
        from models.backbones.video_mae_v2 import Block
        self.num_classes = 0
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=all_frames,
            tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.with_cp = with_cp

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(
                num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                cos_attn=cos_attn) for i in range(depth)
        ])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(
            embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None

        # self.apply(self._init_weights)
        #
        # self.head.weight.data.mul_(init_scale)
        # self.head.bias.data.mul_(init_scale)

        if load_path is not None:
            self.load(load_path)

    def load(self, load_path):
        weight = torch.load(load_path)['module']
        from collections import OrderedDict
        s_state_dict = OrderedDict()
        t_state_dict = self.state_dict()
        for key in weight.keys():
            if key in t_state_dict.keys() and t_state_dict[key].shape == weight[key].shape:
                s_state_dict[key] = weight[key]
        info = self.load_state_dict(s_state_dict, strict=False)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):

        feats = []
        x = self.patch_embed(x)
        B, _, C = x.shape
        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(
                x.device).clone().detach()
        x = self.pos_drop(x)

        x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible

        for blk in self.blocks:
            if self.with_cp:
                x_vis = cp.checkpoint(blk, x_vis)
            else:
                x_vis = blk(x_vis)
            feats.append(x_vis)
        if self.fc_norm is not None:
            return self.fc_norm(x_vis.mean(1))
        else:
            x = self.norm(x_vis)
            return x, feats

    def forward(self, x, mask, **kwargs):
        x, feats = self.forward_features(x, mask)

        return x, feats, None, None


def build_video_mae_s(drop_path_rate=0):
    # encoder = PretrainVisionTransformerEncoder(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
    #                                            drop_rate=0.0,
    #                                            attn_drop_rate=0.0,
    #                                            drop_path_rate=0.1)
    encoder = PreTrainVisionTransformer(
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
    )
    decoder = PretrainVisionTransformerDecoder(
        embed_dim=192,
        num_heads=3
    )
    return encoder, decoder


def build_video_mae_b(drop_path_rate=0):
    encoder = PreTrainVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
        use_mean_pooling=False,
        drop_path_rate=drop_path_rate,
    )
    decoder = PretrainVisionTransformerDecoder()
    return encoder, decoder


class VideoMAEVQA(nn.Module):
    def __init__(self,
                 model_type='s',
                 mask_ratio=0.,
                 head_dropout=0.5,
                 drop_path_rate=0
                 ):
        super(VideoMAEVQA, self).__init__()
        if model_type == 's':
            self.backbone_embed_dim = 384
            self.backbone, self.decoder = build_video_mae_s(drop_path_rate)

        elif model_type == 'b':
            self.backbone_embed_dim = 384 * 2
            self.backbone, self.decoder = build_video_mae_b()

        self.decoder_dim = self.backbone_embed_dim // 2
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None, None], requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None, None], requires_grad=False)
        self.normlize_target = True
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

        self.vqa_head = VQAPoolMlpHead(dropout_ratio=head_dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        self.encoder_to_decoder = nn.Linear(self.backbone_embed_dim, self.decoder_dim,
                                            bias=False)
        # self.encoder_to_cls_decoder = nn.Linear(self.backbone_embed_dim,
        #                                         512, bias=False)

        self.pos_embed = get_sinusoid_encoding_table(self.backbone.pos_embed.shape[1],
                                                     self.decoder_dim)
        self.pos_embed = nn.Parameter(self.pos_embed, requires_grad=False)
        self.fc_norm_mean_pooling = False
        self.masked_patches_type = 'none'
        self.pos_embed_for_cls_decoder = False
        self.mask_token_for_cls_decoder = False
        if self.pos_embed_for_cls_decoder or self.mask_token_for_cls_decoder:
            self.pos_embed_cls = get_sinusoid_encoding_table(self.backbone.pos_embed.shape[1],
                                                             512)
            self.pos_embed_cls = nn.Parameter(self.pos_embed_cls, requires_grad=False)
        if self.mask_token_for_cls_decoder:
            self.mask_token_cls = nn.Parameter(torch.zeros(1, 1, 512))
        if self.fc_norm_mean_pooling:
            self.fc_norm = nn.LayerNorm(self.backbone_embed_dim, eps=1e-6)

        self.mask_ratio = mask_ratio
        if self.mask_ratio <= 0:
            self.decoder = nn.Identity()
            self.encoder_to_decoder = nn.Identity()

    def forward(self, video, mask):
        x_data = video
        mask = mask.bool()
        ####################################
        # new_mask = mask.reshape(10, 1, 8, 14, 14).float()
        # new_mask = new_mask.repeat_interleave(2, dim=2).repeat_interleave(16, dim=3).repeat_interleave(16, dim=4)
        # x_data_mask = x_data * new_mask.to(x_data.device)
        ####################################
        if self.training:
            with torch.no_grad():
                # calculate the predict label
                mean = self.mean.data.clone().detach()
                std = self.std.data.clone().detach()
                unnorm_frames = x_data * std + mean
                t, h, w = unnorm_frames.size(2) // self.tubelet_size, unnorm_frames.size(
                    3) // self.patch_size, unnorm_frames.size(4) // self.patch_size
                if self.normlize_target:
                    images_squeeze = rearrange(unnorm_frames, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                                               p0=self.tubelet_size, p1=self.patch_size, p2=self.patch_size)
                    images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                                   ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                    # we find that the mean is about 0.48 and standard deviation is about 0.08.
                    frames_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
                else:
                    frames_patch = rearrange(unnorm_frames, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
                                             p0=self.tubelet_size, p1=self.patch_size, p2=self.patch_size)
                frames_patch = rearrange(frames_patch, 'b (t s0 h s1 w s2) c -> b (t h w) (s0 s1 s2 c)',
                                         s0=self.mask_stride[0],
                                         s1=self.mask_stride[1],
                                         s2=self.mask_stride[2],
                                         t=t // self.mask_stride[0],
                                         h=h // self.mask_stride[1],
                                         w=w // self.mask_stride[2])
                B, _, C = frames_patch.shape
                labels = frames_patch[(~mask).flatten(1, 2)].reshape(B, -1, C)
        else:
            B = x_data.size(0)
            labels = None
        full_mask = mask.reshape(B, *self.mask_shape).repeat_interleave(self.mask_stride[0], dim=1).repeat_interleave(
            self.mask_stride[1], dim=2).repeat_interleave(self.mask_stride[2], dim=3)
        full_mask = full_mask.flatten(2)
        encoder_logits_backbone, feats, patch_embed, x_vis_list = self.backbone(x_data, ~(full_mask.flatten(1)))
        b, t, p = full_mask.size()
        if self.training:
            pred_pixels = None
            if self.mask_ratio > 0:
                encoder_logits = self.encoder_to_decoder(encoder_logits_backbone)
                c = encoder_logits.size(-1)
                full_mask = full_mask.flatten(1, 2)
                mask_token = self.mask_token.type_as(encoder_logits).repeat(b, t * p, 1)
                mask_token[full_mask, :] = encoder_logits.flatten(0, 1)
                logits_full = mask_token + self.pos_embed.detach().clone()
                pred_pixels = self.decoder(logits_full, -1)
                pred_pixels = rearrange(pred_pixels, 'b (t s0 h s1 w s2) c -> b (t h w) (s0 s1 s2 c)',
                                        s0=self.mask_stride[0],
                                        s1=self.mask_stride[1],
                                        s2=self.mask_stride[2],
                                        t=t // self.mask_stride[0],
                                        h=h // self.mask_stride[1],
                                        w=w // self.mask_stride[2])
                pred_pixels = pred_pixels[(~mask).flatten(1, 2)].reshape(B, -1, C)
        else:
            pred_pixels = None
        preds_score = self.vqa_head(feats)
        output = {"preds_pixel": pred_pixels, "labels_pixel": labels, "preds_score": preds_score, 'feats': feats}
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


class VideoMAEVQAWrapper(BaseModel):
    def __init__(
            self,
            model_type="s",
            mask_ratio=0,
            head_dropout=0.5,
            drop_path_rate=0,
            **kwargs
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.model = VideoMAEVQA(model_type=model_type, mask_ratio=mask_ratio, head_dropout=head_dropout,
                                 drop_path_rate=drop_path_rate)
        self.agent = CellRunningMaskAgent(mask_ratio)

        if model_type == 'b':
            weight = torch.load("/data/ly/code/LinVQATools/pretrained_weights/vit_b_k710_dl_from_giant.pth",
                                map_location='cpu')
            decode_weight = torch.load("/data/ly/code/LinVQATools/pretrained_weights/video_mae_k400.pth",
                                       map_location='cpu')
        elif model_type == 's':
            weight = torch.load("/data/ly/code/LinVQATools/pretrained_weights/vit_s_k710_dl_from_giant.pth",
                                map_location='cpu')
            decode_weight = torch.load("/data/ly/code/LinVQATools/pretrained_weights/video_mae_v1_s_pretrain.pth",
                                       map_location='cpu')
        weight = weight['module']
        t_state_dict = OrderedDict()
        for key in weight.keys():
            weight_value = weight[key]
            key = "model.backbone." + key
            # if 'encoder' in key:
            #     key = key.replace('encoder', 'backbone')
            t_state_dict[key] = weight_value

        weight = decode_weight['model']
        for key in weight.keys():
            if "decoder" in key:
                weight_value = weight[key]
                key = "model." + key
                t_state_dict[key] = weight_value
        t_state_dict = OrderedDict(filter(lambda x: 'encoder_to_decoder' not in x[0], t_state_dict.items()))
        info = self.load_state_dict(t_state_dict, strict=False)
        print(info)

    def forward(self, inputs: torch.Tensor, gt_label=None, data_samples: Optional[list] = None, mode: str = 'tensor',
                **kargs) -> \
            Union[
                Dict[str, torch.Tensor], list]:
        B, Clip, C, D, H, W = inputs.shape
        if mode == 'loss':
            y = gt_label.float().unsqueeze(-1)
            inputs = rearrange(inputs, "b clip c t h w -> (b clip) c t h w")
            self.agent.train()
            mask = self.agent(inputs, [8, 14, 14])['mask']
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
            if self.mask_ratio > 0:
                mae_loss = nn.MSELoss(reduction='none')(output['preds_pixel'], output['labels_pixel']).mean()
                total_loss = mae_loss * 1 + total_loss
                return_dict["total_loss"] = total_loss
                return_dict["mae_lozz"] = mae_loss

            return return_dict
        elif mode == 'predict':
            y = gt_label.float().unsqueeze(-1)
            inputs = rearrange(inputs, "b clip c t h w -> (b clip) c t h w")
            self.agent.eval()
            mask = self.agent(inputs, [8, 14, 14])['mask']
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
            return output['feats']

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

        # losses = {'total_loss': losses['total_loss'], 'vqa_lozz': losses['vqa_lozz'],
        #           'mse_lozz': losses['mse_lozz'], 'mae_lozz': losses['mae_lozz'], 'p_lozz': losses['p_lozz'],
        #           'r_lozz': losses['r_lozz']}
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars


if __name__ == '__main__':
    os.chdir('../')
    weight = torch.load(
        "/data/ly/code/LinVQATools/work_dir/video_mae_vqa/11240027 vit random_cell_mask_75 mae last1/best_SROCC_epoch_358.pth",
        map_location='cpu')["state_dict"]
    t_state_dict = OrderedDict()
    for key in weight.keys():
        if key == "steps":
            continue
        weight_value = weight[key]
        # key = key[7:]
        t_state_dict[key] = weight_value
    model = VideoMAEVQAWrapper(model_type="s", mask_ratio=0.75)
    info = model.load_state_dict(t_state_dict, strict=False)

    video_loader = dict(
        name='FragmentLoader',
        prefix='4frame',
        argument=[
            # dict(
            #     name='FragmentShuffler',
            #     fragment_size=32,
            #     frame_cube=4
            # ),
            # # dict(
            #     name='SpatialShuffler',
            #     fragment_size=32,
            # ),
            dict(
                name='PostProcessSampler',
                num=4,
                frame_cube=4
            )
        ]
    )

    mask = [
        [0,0],
        [0,1]
    ]
    mask = torch.Tensor(mask).repeat(7,7)
    mask = mask.repeat_interleave(16, dim=0).repeat_interleave(16, dim=1)
    mask = mask.reshape(224,224)
    # print(mask.shape)

    visualizer = Visualizer()
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(12, 6))
    left_edge = axes[0][0].get_position().get_points()[0][0]
    fig.text(left_edge - 0.01,
             (axes[0][0].get_position().get_points()[0][1] + axes[0][1].get_position().get_points()[1][1]) / 2,
             f'Example 1', ha='center', va='center', rotation='vertical', fontsize=12)
    left_edge = axes[1][0].get_position().get_points()[0][0]
    fig.text(left_edge - 0.01,
             (axes[1][0].get_position().get_points()[0][1] + axes[1][1].get_position().get_points()[1][1]) / 2,
             f'Example 2', ha='center', va='center', rotation='vertical', fontsize=12)
    left_edge = axes[2][0].get_position().get_points()[0][0]
    fig.text(left_edge - 0.01,
             (axes[2][0].get_position().get_points()[0][1] + axes[2][1].get_position().get_points()[1][1]) / 2,
             f'Example 3', ha='center', va='center', rotation='vertical', fontsize=12)
    # 为最下面的子图添加标签
    for j, ax in enumerate(axes[2]):
        ax.text(0.5, -0.15, f"{(j+1)*2}", ha='center', va='center', transform=ax.transAxes)

    ax_title = fig.add_subplot(111, frame_on=False)
    ax_title.set_xticks([])
    ax_title.set_yticks([])
    ax_title.set_frame_on(False)
    ax_title.text(0.5, 1.05, 'Multi Stage Attention', ha='center', va='center', fontsize=16,
                  fontweight='bold')

    dataset = SingleBranchDataset(video_loader=video_loader, norm=True, clip=1)

    data = dataset[0]
    inputs = data['inputs'].unsqueeze(0)
    raw_video = data['raw_video'].unsqueeze(0)
    feats = model(inputs, gt_label=torch.rand((2)), mode='tensor')

    for j,feat in enumerate(feats[1::2]):
        feat = feat.reshape(8, 7, 7, 384).detach().cpu().permute(3, 0, 1, 2)
        img = raw_video[0,0,:,0,...].detach().cpu().permute(1, 2, 0).numpy().astype('uint8')
        mask = mask.bool()
        img = img[mask].reshape(112,112,3)
        print(img.shape)
        drawn_img = visualizer.draw_featmap(feat[:, 0, ...],img, channel_reduction='select_max')
        # 绘制小图
        axes[0, j].imshow(drawn_img, cmap='viridis')  # 可以根据需要设置不同的颜色映射
        axes[0, j].axis('off')  # 关闭坐标轴

    data = dataset[50]
    inputs = data['inputs'].unsqueeze(0)
    raw_video = data['raw_video'].unsqueeze(0)
    feats = model(inputs, gt_label=torch.rand((2)), mode='tensor')

    for j, feat in enumerate(feats[1::2]):
        feat = feat.reshape(8, 7, 7, 384).detach().cpu().permute(3, 0, 1, 2)
        img = raw_video[0, 0, :, 0, ...].detach().cpu().permute(1, 2, 0).numpy().astype('uint8')
        mask = mask.bool()
        img = img[mask].reshape(112, 112, 3)
        print(img.shape)
        drawn_img = visualizer.draw_featmap(feat[:, 0, ...], img, channel_reduction='select_max')
        # 绘制小图
        axes[1, j].imshow(drawn_img, cmap='viridis')  # 可以根据需要设置不同的颜色映射
        axes[1, j].axis('off')  # 关闭坐标轴

    data = dataset[200]
    inputs = data['inputs'].unsqueeze(0)
    raw_video = data['raw_video'].unsqueeze(0)
    feats = model(inputs, gt_label=torch.rand((2)), mode='tensor')

    for j, feat in enumerate(feats[1::2]):
        feat = feat.reshape(8, 7, 7, 384).detach().cpu().permute(3, 0, 1, 2)
        img = raw_video[0, 0, :, 0, ...].detach().cpu().permute(1, 2, 0).numpy().astype('uint8')
        mask = mask.bool()
        img = img[mask].reshape(112, 112, 3)
        print(img.shape)
        drawn_img = visualizer.draw_featmap(feat[:, 0, ...], img, channel_reduction='select_max')
        # 绘制小图
        axes[2, j].imshow(drawn_img, cmap='viridis')  # 可以根据需要设置不同的颜色映射
        axes[2, j].axis('off')  # 关闭坐标轴
    # 调整布局
    plt.subplots_adjust(hspace=0.2, wspace=0.1)

    # 显示图形
    plt.show()