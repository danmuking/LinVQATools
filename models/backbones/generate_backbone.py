import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

from models import logger
from models.backbones.swin_backbone import PatchEmbed3D, BasicLayer, PatchMerging, get_adaptive_window_size


class SwinTransformer3D(nn.Module):
    """Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(
            self,
            pretrained=None,
            pretrained2d=False,
            patch_size=(2, 4, 4),
            in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=(8, 7, 7),
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            frozen_stages=-1,
            use_checkpoint=True,
            jump_attention=[False, False, False, False],
            frag_biases=[True, True, True, False],
            base_x_size=(32, 224, 224),
            load_path=None
    ):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        # 编码维度
        self.embed_dim = embed_dim
        # patch是否使用norm
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        # 输入维度
        self.base_x_size = base_x_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer]
                if isinstance(window_size, list)
                else window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
                jump_attention=jump_attention[i_layer],
                frag_bias=frag_biases[i_layer],
            )
            self.layers.append(layer)

        self.vqa_feature = BasicLayer(
            dim=int(embed_dim * 2 ** (self.num_layers - 1)),
            depth=depths[(self.num_layers - 1)],
            num_heads=num_heads[(self.num_layers - 1)],
            window_size=window_size[(self.num_layers - 1)]
            if isinstance(window_size, list)
            else window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:(self.num_layers - 1)]): sum(depths[: (self.num_layers - 1) + 1])],
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            jump_attention=jump_attention[(self.num_layers - 1)],
            frag_bias=frag_biases[(self.num_layers - 1)],
        )
        self.generate_feature = BasicLayer(
            dim=int(embed_dim * 2 ** (self.num_layers - 1)),
            depth=depths[(self.num_layers - 1)],
            num_heads=num_heads[(self.num_layers - 1)],
            window_size=window_size[(self.num_layers - 1)]
            if isinstance(window_size, list)
            else window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:(self.num_layers - 1)]): sum(depths[: (self.num_layers - 1) + 1])],
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            jump_attention=jump_attention[(self.num_layers - 1)],
            frag_bias=frag_biases[(self.num_layers - 1)],
        )
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)

        self._freeze_stages()

        self.init_weights()

        if load_path is not None:
            self.load(load_path)

    def load(self, load_path):
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
            t_state_dict = self.state_dict()
            for key, value in t_state_dict.items():
                if key in i_state_dict and i_state_dict[key].shape != value.shape:
                    i_state_dict.pop(key)
            # print(i_state_dict.keys())
            info = self.load_state_dict(i_state_dict, strict=False)
            logger.info("faster vqa swin加载{}权重,info:{} ".format(load_path, info))

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location="cpu")
        state_dict = checkpoint["model"]

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [
            k for k in state_dict.keys() if "relative_position_index" in k
        ]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict["patch_embed.proj.weight"] = (
                state_dict["patch_embed.proj.weight"]
                .unsqueeze(2)
                .repeat(1, 1, self.patch_size[0], 1, 1)
                / self.patch_size[0]
        )

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [
            k for k in state_dict.keys() if "relative_position_bias_table" in k
        ]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            wd = self.window_size[0]
            if nH1 != nH2:
                print(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = (
                        torch.nn.functional.interpolate(
                            relative_position_bias_table_pretrained.permute(1, 0).view(
                                1, nH1, S1, S1
                            ),
                            size=(
                                2 * self.window_size[1] - 1,
                                2 * self.window_size[2] - 1,
                            ),
                            mode="bicubic",
                        )
                    )
                    relative_position_bias_table_pretrained = (
                        relative_position_bias_table_pretrained_resized.view(
                            nH2, L2
                        ).permute(1, 0)
                    )
            state_dict[k] = relative_position_bias_table_pretrained.repeat(
                2 * wd - 1, 1
            )

        msg = self.load_state_dict(state_dict, strict=False)
        print(msg)
        print(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def load_swin(self, load_path, strict=False):
        print("loading swin lah")
        from collections import OrderedDict

        model_state_dict = self.state_dict()
        state_dict = torch.load(load_path)["state_dict"]

        clean_dict = OrderedDict()
        for key, value in state_dict.items():
            if "backbone" in key:
                clean_key = key[9:]
                clean_dict[clean_key] = value
                if "relative_position_bias_table" in clean_key:
                    forked_key = clean_key.replace(
                        "relative_position_bias_table", "fragment_position_bias_table"
                    )
                    if forked_key in clean_dict:
                        print("load_swin_error?")
                    else:
                        clean_dict[forked_key] = value

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [
            k for k in clean_dict.keys() if "relative_position_bias_table" in k
        ]
        for k in relative_position_bias_table_keys:
            print(k)
            relative_position_bias_table_pretrained = clean_dict[k]
            relative_position_bias_table_current = model_state_dict[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if isinstance(self.window_size, list):
                i_layer = int(k.split(".")[1])
                L2 = (2 * self.window_size[i_layer][1] - 1) * (
                        2 * self.window_size[i_layer][2] - 1
                )
                wd = self.window_size[i_layer][0]
            else:
                L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
                wd = self.window_size[0]
            if nH1 != nH2:
                print(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int((L1 / 15) ** 0.5)
                    print(
                        relative_position_bias_table_pretrained.shape, 15, nH1, S1, S1
                    )
                    relative_position_bias_table_pretrained_resized = (
                        torch.nn.functional.interpolate(
                            relative_position_bias_table_pretrained.permute(1, 0)
                            .view(nH1, 15, S1, S1)
                            .transpose(0, 1),
                            size=(
                                2 * self.window_size[i_layer][1] - 1,
                                2 * self.window_size[i_layer][2] - 1,
                            ),
                            mode="bicubic",
                        )
                    )
                    relative_position_bias_table_pretrained = (
                        relative_position_bias_table_pretrained_resized.transpose(
                            0, 1
                        ).view(nH2, 15, L2)
                    )
            clean_dict[k] = relative_position_bias_table_pretrained  # .repeat(2*wd-1,1)

        ## Clean Mismatched Keys
        for key, value in model_state_dict.items():
            if key in clean_dict:
                if value.shape != clean_dict[key].shape:
                    print(key)
                    clean_dict.pop(key)

        self.load_state_dict(clean_dict, strict=strict)

    def init_weights(self, pretrained=None):
        print(self.pretrained, self.pretrained2d)
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger()
            # logger.info(f"load model from: {self.pretrained}")

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights()
            else:
                # Directly load 3D model.
                self.load_swin(self.pretrained, strict=False)  # , logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x, need_feat=False, multi=False, layer=-1, adaptive_window_size=True):

        """Forward function."""
        # 生成自适应窗口大小
        if adaptive_window_size:
            # 8,7,7
            resized_window_size = get_adaptive_window_size(self.window_size, x.shape[2:], self.base_x_size)
        else:
            resized_window_size = None

        # 将视频划分为不重叠的patch
        x = self.patch_embed(x)

        # dropout
        x = self.pos_drop(x)
        feats = [x]

        for l, mlayer in enumerate(self.layers):
            x = mlayer(x.contiguous(), resized_window_size)
            feats += [x]

        vqa_feat = self.vqa_feature(x)
        generate_feat = self.generate_feature(x)

        vqa_feat = rearrange(vqa_feat, "n c d h w -> n d h w c")
        vqa_feat = self.norm(vqa_feat)
        vqa_feat = rearrange(vqa_feat, "n d h w c -> n c d h w")

        generate_feat = rearrange(generate_feat, "n c d h w -> n d h w c")
        generate_feat = self.norm(generate_feat)
        generate_feat = rearrange(generate_feat, "n d h w c -> n c d h w")

        if multi:
            shape = x.shape[2:]
            return torch.cat([F.interpolate(xi, size=shape, mode="trilinear") for xi in feats[:-1]], 1)
        elif layer > -1:
            print("something", len(feats))
            return feats[layer]
        elif need_feat:
            return feats
        else:
            return tuple([[vqa_feat, generate_feat]])

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer3D, self).train(mode)
        self._freeze_stages()
