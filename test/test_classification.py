from unittest import TestCase

import torch

from models.heads.classification import ClassificationHead


class TestClassificationHead(TestCase):
    def test(self):
        model = ClassificationHead()
        x = torch.zeros((2, 768, 8, 14, 14))
        y = model([[x]])
        print(y.shape)
