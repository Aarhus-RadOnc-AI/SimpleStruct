import unittest

import SimpleITK as sitk
import numpy as np

from simplestruct.metrics.axis_difference import AxisDiff
from simplestruct.metrics.surface_dice import SurfaceDice
from tests.utils import load_ref_and_pred


class TestAxisDiff(unittest.TestCase):
    def test_axis_diff(self):
        ref = np.array([
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]],
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]]
        ])
        pred = np.array([
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]],
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]],
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
        ])

        ad = AxisDiff(ref, pred)
        ad.execute()
        self.assertDictEqual(ad.get_axis_difference(), {0: {'min': -1, 'max': 1}, 1: {'min': -1, 'max': -1}, 2: {'min': -1, 'max': -1}})
        self.assertDictEqual(ad.get_axis_difference(1), {'min': -1, 'max': -1})
        print(ad.get_axis_difference(1))


if __name__ == '__main__':
    unittest.main()
