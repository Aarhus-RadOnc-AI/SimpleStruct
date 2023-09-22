import unittest

import SimpleITK as sitk
import numpy as np

from simplestruct.metrics.surface_dice import SurfaceDice
from tests.utils import load_ref_and_pred


class TestSurfaceDice(unittest.TestCase):
    def setUp(self) -> None:
        self.ref_img, self.other_img = load_ref_and_pred()

    def test_surface_dice(self):
        ref = np.array([
            [[0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0]]
        ])
        pred = np.array([
            [[0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 0],
             [0, 1, 0, 0, 1, 0],
             [0, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0]]
        ])
        ref_img = sitk.GetImageFromArray(ref)
        pred_img = sitk.GetImageFromArray(pred)
        surface_dice = SurfaceDice(ref_img, pred_img)
        self.assertEqual(surface_dice.get_surface_dice(0.5), 0.875)
        self.assertEqual(surface_dice.get_surface_dice(1), 1)
        self.assertEqual(surface_dice.get_surface_dice(3), 1)


if __name__ == '__main__':
    unittest.main()
