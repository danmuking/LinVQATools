import os
from unittest import TestCase

import torch

from data.sampler.spatial_fragment_sampler import PlaneSpatialFragmentSampler,SphereSpatialFragmentSampler


class TestPlaneSpatialFragmentSampler(TestCase):
    def test_plane_patial_fragment_sampler(self):
        os.chdir('../')
        video = torch.randn((3,32,1920,1080))
        sampler = PlaneSpatialFragmentSampler()
        sampler(video)

    def test_sphere_patial_fragment_sampler(self):
        os.chdir('../')
        video = torch.randn((3,32,1920,1080))
        sampler = SphereSpatialFragmentSampler
        sampler(video)
