import torch
import torch.nn as nn
from functools import partial, reduce

from .backbones.base_swin_backbone import SwinTransformer3D
from .backbones.mvit import MViT
from .backbones.swin_backbone import SwinTransformer3D as VideoBackbone
from .backbones.video_mae_v2 import VisionTransformer
import models.heads as heads
from .utils.blur import get_blur_net, get_blur_vec


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
        self.blur = 4
        if backbone == 'faster_vqa':
            b = VideoBackbone(
                base_x_size=base_x_size,
                window_size=window_size,
                load_path=load_path
            )
        elif backbone == 'mvit':
            b = MViT(arch='tiny', drop_path_rate=drop_path_rate)
            b.init_weights()
        elif backbone == 'swin':
            b = SwinTransformer3D(arch='tiny')
        elif backbone == 'vit':
            b = VisionTransformer(
                patch_size=16,
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
        # self.vqa_head = getattr(heads, vqa_head['name'])(**vqa_head)

        self.deblur_net = get_blur_net()
        state_dict = torch.load('pretrained_weights/Stripformer_realblur_J.pth',map_location='cpu')
        from collections import OrderedDict
        i_state_dict = OrderedDict()
        for key in state_dict.keys():
            t_key = key.replace("module.", "")
            i_state_dict[t_key] = state_dict[key]
        self.deblur_net.load_state_dict(i_state_dict)
        self.deblur_net = self.deblur_net.eval()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = self.regression(3072+1280,128)

    def regression(self,in_channel,hid_channel,drop_rate=0.5):
        return nn.Sequential(
            nn.Linear(in_channel,in_channel),
            nn.LayerNorm(in_channel,eps=1e-6),
            nn.GELU(),
            nn.Dropout(p=0.5) if 0.5 > 0 else nn.Identity(),
            nn.Linear(in_channel, hid_channel),
            nn.GELU(),
            nn.Dropout(p=0.5) if 0.5 > 0 else nn.Identity(),
            nn.Linear(hid_channel, 1),
            nn.GELU(),
        )
    def forward(self, vclips, inference=False, return_pooled_feats=False, reduce_scores=True, pooled=False, **kwargs):
        b,c,d,h,w = vclips.shape
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
                    feat = torch.flatten(self.avg_pool(feat[0][0]), 1)

                    blur_feats = get_blur_vec(self.deblur_net, vclips[key], self.blur)
                    blur_feats = torch.flatten(self.avg_pool(blur_feats), 1)
                    blur_feats = blur_feats.reshape(b, self.blur * blur_feats.size(1))
                    feat = torch.cat([feat, blur_feats], dim=1)
                    scores += [self.linear(feat)]
                    # scores += [getattr(self, "vqa_head")(feat)]
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
                feat = torch.flatten(self.avg_pool(feat[0][0]),1)


                blur_feats = get_blur_vec(self.deblur_net, vclips[key], self.blur)
                blur_feats = torch.flatten(self.avg_pool(blur_feats), 1)
                blur_feats = blur_feats.reshape(b, self.blur * blur_feats.size(1))
                feat = torch.cat([feat, blur_feats], dim=1)
                scores += [self.linear(feat)]
                # scores += [getattr(self, "vqa_head")(feat)]
            return scores
