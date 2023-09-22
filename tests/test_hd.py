import os
import unittest

import SimpleITK as sitk

from simplestruct.metrics.hd import HD
from tests.utils import get_test_dicom, load_ref_and_pred


class TestHd(unittest.TestCase):
    def setUp(self) -> None:
        self.ref_img, self.other_img = load_ref_and_pred()

    def test_hd(self):
        f = sitk.HausdorffDistanceImageFilter()
        f.Execute(self.ref_img, self.other_img)
        ref_hd = f.GetHausdorffDistance()
        hd = HD(self.ref_img, self.other_img)
        self.assertEqual(ref_hd, hd.get_max_min_hd())

if __name__ == '__main__':
    unittest.main()
