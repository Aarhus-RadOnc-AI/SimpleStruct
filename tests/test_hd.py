import unittest
import SimpleITK as sitk

from simplestruct.metrics.distance.hd import HD
from simplestruct.single_filters.get_edge_of_structure import get_edge_of_structure
import numpy as np
class TestHd(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_path = "test_data/HN1004_20190403_CT/scans/1-unknown/resources/NIFTI/files/mask_Spinal-Cord.nii.gz"
        self.other_path = "test_data/test/Spinal-Cord.nii.gz"
        self.ref_img = sitk.ReadImage(self.reference_path)
        self.other_img = sitk.ReadImage(self.other_path)

  #      self.ref = get_edge_of_structure(sitk.GetArrayFromImage(self.ref_img), 255)
  #      self.other = get_edge_of_structure(sitk.GetArrayFromImage(self.other_img), 1)

        #self.distance_arr = get_distance_matrix(self.ref, self.other, tuple(reversed(self.ref_img.GetSpacing())))

    def test_init_HD(self):
        hd = HD(self.ref_img, self.other_img, label_int=(255, 1))
        hd.set_distance_matrix()
        print(hd.distance_matrix)


    # def test_hd(self):
    #     f = sitk.HausdorffDistanceImageFilter()
    #     f.Execute(self.ref_img, self.other_img)
    #     ref_hd = f.GetHausdorffDistance()
    #     print(ref_hd)
    #
    #     arr = get_distance_matrix(self.ref, self.other, tuple(reversed(self.ref_img.GetSpacing())))
    #     other_hd = arr.min(axis=1).max()
    #     print(other_hd)
    #     self.assertEqual(ref_hd, other_hd)


if __name__ == '__main__':
    unittest.main()
