import os
import unittest
from simplestruct.structure_set.structure_set import StructureSet
import SimpleITK as sitk
from simplestruct.metrics import overlap


class TestStructureSet(unittest.TestCase):
    def setUp(self) -> None:
        self.nifti_image = "test_data/HN1004_20190403_CT/scans/1_3_6_1_4_1_40744_29_33371661027192187491509798061184654147-unknown/resources/NIFTI/files/image.nii.gz"
        self.dicom_image = "test_data/HN1004_20190403_CT/scans/1_3_6_1_4_1_40744_29_33371661027192187491509798061184654147-unknown/resources/DICOM/files/"
        self.rtstruct = "test_data/HN1004_20190403_CT/scans/1-unknown/resources/secondary/files/1.3.6.1.4.1.40744.29.60478521633161295138880069450542725477-1-1-1mxpuet.dcm"
        self.nifti_folder = "test_data/HN1004_20190403_CT/scans/1-unknown/resources/NIFTI/files"
    def test_parse_dicom_image(self):
        dicom_image = StructureSet.parse_dicom_image(self.dicom_image)
        ss = StructureSet(reference_image=dicom_image)
        self.assertIsInstance(ss.reference_image, sitk.Image)

    def test_parse_rtstruct(self):
        dicom_image = StructureSet.parse_dicom_image(self.dicom_image)
        ss = StructureSet(reference_image=dicom_image)
        ss.from_dicom_rtstruct(self.rtstruct)
        for label in ss.list_structure_names():
            mask = ss.get_structure(label)
            for file in os.listdir(self.nifti_folder):
                if label in file:
                    other_mask = sitk.ReadImage(os.path.join(self.nifti_folder, file))
                    other_mask = other_mask == 255
                    self.assertGreater(overlap.calculate_dice(mask, other_mask), 0.85)
                    break

if __name__ == '__main__':
    unittest.main()
