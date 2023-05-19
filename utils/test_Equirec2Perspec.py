from unittest import TestCase

import cv2

from Equirec2Perspec import Equirectangular


class TestEquirectangular(TestCase):
    def test_get_perspective(self):
        equ = Equirectangular('/tmp/pycharm_project_397/test2.jpg')  # Load equirectangular image

        #
        # FOV unit is degree 
        # theta is z-axis angle(right direction is positive, left direction is negative)
        # phi is y-axis angle(up direction positive, down direction negative)
        # height and width is output image dimension 
        #
        img = equ.GetPerspective(60 , 0, -90, 32, 32)  # Specify parameters(FOV, theta, phi, height, width)
        cv2.imwrite("test.jpg",img)
