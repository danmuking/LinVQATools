import os
import time
from unittest import TestCase

import cv2
import numpy as np
import torch

from data.sampler.spatial_fragment_sampler import PlaneSpatialFragmentSampler, SphereSpatialFragmentSampler, \
    FastPlaneSpatialFragmentSampler


class TestPlaneSpatialFragmentSampler(TestCase):
    def test_plane_patial_fragment_sampler(self):
        os.chdir('../')
        video = torch.randn((3, 32, 1920, 1080))
        sampler = PlaneSpatialFragmentSampler(aligned=8)
        start_time = time.time()
        sampler(video)
        print(time.time() - start_time)

    def test_sphere_patial_fragment_sampler(self):
        os.chdir('../')
        video = torch.randn((3, 32, 1920, 1080))
        sampler = SphereSpatialFragmentSampler()
        sampler(video)

    def test_fast_plane_patial_fragment_sampler(self):
        os.chdir('../')
        video = cv2.imread('./test.jpg')
        video = video.transpose(2,0,1)[:,None,:,:]
        video = torch.from_numpy(video)
        video = torch.randn((3, 32, 1920, 1080))
        # print(video.shape)
        sampler = FastPlaneSpatialFragmentSampler(aligned=8)
        start_time = time.time()
        sampler(video)
        print(time.time() - start_time)