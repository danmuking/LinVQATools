import torch
import torch.nn as nn
from functools import partial, reduce

from .backbones.base_swin_backbone import SwinTransformer3D
from .backbones.mvit import MViT
from .backbones.swin_backbone import SwinTransformer3D as VideoBackbone
from .backbones.video_mae_v2 import VisionTransformer
import models.heads as heads
# from .neck.patch_weighted import PatchWeighted


class DiViDeAddEvaluator(nn.Module):
    def __init__(
            self,
            window_size=(8, 7, 7),
            multi=False,
            layer=-1,
            backbone='swin',
            base_x_size=(32, 224, 224),
            vqa_head=dict(name='VQAHead', in_channels=768),
            drop_path_rate=0.2,
            load_path=None
    ):
        super().__init__()
        self.multi = multi
        self.layer = layer
        if backbone == 'faster_vqa':
            b = VideoBackbone(
                base_x_size=base_x_size,
                window_size=window_size,
                load_path=load_path,
                # drop_path_rate=0.2,
            )
        elif backbone == 'mvit':
            b = MViT(
                arch='small',
                drop_path_rate=drop_path_rate,
                load_path=load_path,
            )
        elif backbone == 'swin':
            b = SwinTransformer3D(arch='tiny')
        elif backbone == 'vit':
            b = VisionTransformer(
                patch_size=8,
                embed_dim=384,
                depth=12,
                num_heads=6,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                load_path=load_path,
                num_classes=0,
                use_mean_pooling=False
            )
        print("Setting backbone:", 'fragments' + "_backbone")
        setattr(self, 'fragments' + "_backbone", b)

        # self.neck = PatchWeighted()
        self.vqa_head = getattr(heads, vqa_head['name'])(**vqa_head)
        # self.motion_head = getattr(heads, vqa_head['name'])(fc_out=2,**vqa_head)

    def forward(self, vclips, inference=False, return_pooled_feats=False, reduce_scores=True, pooled=False, **kwargs):
        vclips = {
            'fragments': vclips
        }
        if inference:
            self.eval()
            with torch.no_grad():

                scores = []
                feats = {}
                for key in vclips:
                    # key = 'fragments'
                    feat = getattr(self, key.split("_")[0] + "_backbone")(vclips[key], multi=self.multi,
                                                                          layer=self.layer, **kwargs)
                    # feat = self.neck(feat)
                    scores += [getattr(self, "vqa_head")(feat)]
                    # scores += [getattr(self, "motion_head")(feat)]
            self.train()
            return scores
        else:
            self.train()
            scores = []
            feats = {}
            for key in vclips:
                # key = 'fragments_backbone'
                feat = getattr(self, key.split("_")[0] + "_backbone")(vclips[key], multi=self.multi, layer=self.layer,
                                                                      **kwargs)
                # feat = self.neck(feat)
                scores += [getattr(self, "vqa_head")(feat)]
                # scores += [getattr(self, "motion_head")(feat)]
            return scores
