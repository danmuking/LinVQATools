import os
from unittest import TestCase

import torch

from data.shuffler.fragment_shuffler import FragmentShuffler


class TestFragmentShuffler(TestCase):
    def test_shuffle(self):
        os.chdir('../')
        shuffler = FragmentShuffler()
        data = torch.zeros((3,32,224,224))
        shuffler.shuffle(data)
