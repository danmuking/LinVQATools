from unittest import TestCase

from data.loader.sampler_loader import VideoSamplerLoader


class TestVideoSamplerLoader(TestCase):
    def test(self):
        argument = [
            dict(
                name='FragmentShuffler',
            ),
            dict(
                name='PostProcessSampler',
                num=2
            )
        ]
        loader = VideoSamplerLoader(argument=argument)
        loader('/data/ly/VQA_ODV/Group1/G1AbandonedKingdom_TSP_7680x3840_fps30_qp42_2878k_ERP.mp4',300)
