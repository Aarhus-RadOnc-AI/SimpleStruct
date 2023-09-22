import os
import unittest

import SimpleITK as sitk

from simplestruct.metrics import DescriptiveStats
from simplestruct.metrics.hd import HD
from tests.utils import load_ref_and_pred

class TestDescriptiveStats(unittest.TestCase):
    def setUp(self) -> None:
        self.ref_img, self.other_img = load_ref_and_pred()

    def test_descriptive_stats(self):
        ds = DescriptiveStats(self.ref_img)
        ds.execute()
        print(ds.get_descriptive_stats())
        self.assertIsNotNone(ds.get_descriptive_stats())
        self.assertIsInstance(ds.get_descriptive_stats(), dict)


if __name__ == '__main__':
    unittest.main()
