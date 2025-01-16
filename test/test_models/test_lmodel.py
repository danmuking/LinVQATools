from unittest import TestCase

import torch
from torch import nn

from models.last_model import LModel, LModelWrapper
from models.model import Model, ModelWrapper
from models.video_mae_vqa import CellRunningMaskAgent


class TestCellRunningMaskAgent(TestCase):
    def test_LModel(self):
        agent = CellRunningMaskAgent(0.5)
        model = LModel(mask_ratio=0.75).cuda()
        inputs = {"video":torch.rand((2, 3, 16, 224, 224)).cuda()}
        agent.train()
        mask = agent(inputs, [8, 14, 14])['mask']
        mask = mask.reshape(mask.size(0), 8, -1).cuda()
        model(inputs,mask)
    def test_lmodel_wrapper(self):
        model = LModelWrapper(mask_ratio=0.25)
        inputs = {"video":torch.rand((6,1, 3, 16, 224, 224)),"tem_img":torch.rand((6,1,3,160,160)),"spa_img":torch.rand((6,1,3,224,224))}
        y = model(inputs,gt_label=torch.rand((6)),gt_class=torch.empty(6, dtype=torch.long).random_(5),mode='loss')
        print(y)