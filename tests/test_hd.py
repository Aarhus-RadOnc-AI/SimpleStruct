import unittest

import SimpleITK as sitk

from simplestruct.metrics.hd import HD


class TestHd(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_path = "test_data/HN1004_20190403_CT/scans/1-unknown/resources/NIFTI/files/mask_Spinal-Cord.nii.gz"
        self.other_path = "test_data/test/Spinal-Cord.nii.gz"
        self.ref_img = sitk.ReadImage(self.reference_path)
        self.other_img = sitk.ReadImage(self.other_path)

    def test_hd(self):
        f = sitk.HausdorffDistanceImageFilter()
        f.Execute(self.ref_img, self.other_img)
        ref_hd = f.GetHausdorffDistance()
        hd = HD(self.ref_img, self.other_img, label_int=(255, 1))
        self.assertEqual(ref_hd, hd.get_max_min_hd())

if __name__ == '__main__':
    unittest.main()
