from unittest import TestCase

import torch

from models.backbones.swin_backbone import SwinTransformer3D


class TestVideoSwinTransformer(TestCase):
    def test(self):
        model = SwinTransformer3D(base_x_size=(16, 224, 224))
        load_path = '/home/ly/code/LinVQATools/pretrained_weights/swin_tiny_patch244_window877_kinetics400_1k.pth'
        model.load_weight(load_path)
        # x = torch.zeros((1, 3, 16, 224, 224))
        # feature = model(x)

        # print([f.shape for f in feature])
        # state_dict = torch.load(load_path, map_location='cpu')
        # print(state_dict["state_dict"].keys())

        # if "state_dict" in state_dict:
        #     ### migrate training weights from mmaction
        #     state_dict = state_dict["state_dict"]
        #     from collections import OrderedDict
        #
        #     i_state_dict = OrderedDict()
        #     for key in state_dict.keys():
        #         if "head" in key:
        #             continue
        #         if "cls" in key:
        #             tkey = key.replace("cls", "vqa")
        #         elif "backbone" in key:
        #             # i_state_dict[key] = state_dict[key]
        #             i_state_dict["fragments_" + key] = state_dict[key]
        #             # i_state_dict["resize_" + key] = state_dict[key]
        #         else:
        #             i_state_dict[key] = state_dict[key]
        #     t_state_dict = self.model.state_dict()
        #     for key, value in t_state_dict.items():
        #         if key in i_state_dict and i_state_dict[key].shape != value.shape:
        #             i_state_dict.pop(key)
        #     info = self.model.load_state_dict(i_state_dict, strict=False)
        #     logger.info("权重加载完成,info:{}".format(info))
        # print(model.state_dict().keys())