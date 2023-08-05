from unittest import TestCase

from data.sampler.time_fragment_sampler import CubeExtractSample


class TestCubeExtractSample(TestCase):
    def test(self):
        sampler = CubeExtractSample(3)
        print(sampler(6))
