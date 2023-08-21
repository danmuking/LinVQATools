from unittest import TestCase

from data.loader import FragmentLoader


class TestFragmentLoader(TestCase):
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
        loader = FragmentLoader(
            argument=argument
        )
        video = loader('/data/ly/VQA_ODV/Group1/G1AbandonedKingdom_TSP_7680x3840_fps30_qp42_2878k_ERP.mp4')
        print(video.shape)
