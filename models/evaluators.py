import torch
import torch.nn as nn
from functools import partial, reduce

from .backbones.base_swin_backbone import SwinTransformer3D
from .backbones.mvit import MViT
from .backbones.swin_backbone import SwinTransformer3D as VideoBackbone

from .heads.head import VQAHead


class DiViDeAddEvaluator(nn.Module):
    def __init__(
            self,
            window_size=(8,7,7),
            multi=False,
            layer=-1,
            backbone='swin',
            base_x_size=(32, 224, 224),
            vqa_head=dict(in_channels=768),
            drop_path_rate=0.2
    ):
        super().__init__()
        self.multi = multi
        self.layer = layer
        if backbone == 'faster_vqa':
            b = VideoBackbone(
                base_x_size=base_x_size,
                window_size=window_size
            )
        elif backbone == 'mvit':
            b = MViT(arch='tiny', drop_path_rate=drop_path_rate)
            b.init_weights()
        elif backbone =='swin':
            b = SwinTransformer3D(arch='tiny')
        print("Setting backbone:", 'fragments' + "_backbone")
        setattr(self, 'fragments' + "_backbone", b)
        self.vqa_head = VQAHead(**vqa_head)

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
                    scores += [getattr(self, "vqa_head")(feat)]
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
                scores += [getattr(self, "vqa_head")(feat)]
            return scores
