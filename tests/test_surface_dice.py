import unittest

import SimpleITK as sitk

from simplestruct.metrics.distance.hd import HD
from simplestruct.metrics.exotic.surface_dice import SurfaceDice


class TestSurfaceDice(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_path = "test_data/HN1004_20190403_CT/scans/1-unknown/resources/NIFTI/files/mask_Spinal-Cord.nii.gz"
        self.other_path = "test_data/test/Spinal-Cord.nii.gz"
        self.ref_img = sitk.ReadImage(self.reference_path)
        self.other_img = sitk.ReadImage(self.other_path)

    def test_surface_dice(self):

        surface_dice = SurfaceDice(self.ref_img, self.other_img, label_int=(255, 1))
        print(surface_dice.get_surface_dice(0.5))
        print(surface_dice.get_surface_dice(1))
        print(surface_dice.get_surface_dice(3))

if __name__ == '__main__':
    unittest.main()
