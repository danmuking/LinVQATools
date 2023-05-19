import os
from unittest import TestCase

from data.sampler import FragmentSampleFrames


class TestFragmentSampleFrames(TestCase):
    def test_get_frame_indices(self):
        os.chdir('../')
        sampler = FragmentSampleFrames(32 // 8, 8, 2, 1)
        frame_inds = sampler(300)
        print(frame_inds)
