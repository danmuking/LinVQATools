from unittest import TestCase

import torch

from models.model import Model, ModelWrapper
from models.video_mae_vqa import CellRunningMaskAgent


class TestCellRunningMaskAgent(TestCase):
    def test_Model(self):
        agent = CellRunningMaskAgent(0.75)
        model = Model()
        inputs = {"video":torch.rand((2, 3, 16, 224, 224)),"img":torch.rand((2,3,224,224))}
        agent.train()
        mask = agent(inputs, [8, 14, 14])['mask']
        mask = mask.reshape(mask.size(0), 8, -1)
        model(inputs,mask)

    def test_model_wrapper(self):
        model = ModelWrapper()
        inputs = {"video":torch.rand((3,2, 3, 16, 224, 224)),"img":torch.rand((3,2,3,224,224))}
        y = model(inputs,gt_label=torch.rand((3)),mode='predict')
        print(y)