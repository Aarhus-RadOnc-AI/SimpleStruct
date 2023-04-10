import unittest

import SimpleITK as sitk

from simplestruct.metrics.distance.hd import HD
from simplestruct.multi_filters.hausdorff_map import generate_structure_hausdorff_map


class TestHausdorffMap(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_path = "test_data/HN1004_20190403_CT/scans/1-unknown/resources/NIFTI/files/mask_Spinal-Cord.nii.gz"
        self.other_path = "test_data/test/Spinal-Cord.nii.gz"
        self.ref_img = sitk.ReadImage(self.reference_path)
        self.other_img = sitk.ReadImage(self.other_path)

    def test_hausdorff_map(self):
        hd = generate_structure_hausdorff_map(self.ref_img == 255,
                                              [self.other_img == 1, self.other_img == 1, self.ref_img == 255])


if __name__ == '__main__':
    unittest.main()
