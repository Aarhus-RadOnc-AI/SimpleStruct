import os
from typing import Dict

import SimpleITK as sitk
import numpy as np

from . import parser, converter


class StructureSet:
    def __init__(self, reference_image: sitk.Image):
        self.reference_image = reference_image
        self.structures: Dict[str, sitk.Image] = {}

    @staticmethod
    def parse_dicom_image(path):
        dicom_reader = sitk.ImageSeriesReader()
        dicom_file_names = dicom_reader.GetGDCMSeriesFileNames(str(path))
        dicom_reader.SetFileNames(dicom_file_names)
        dicom_image = dicom_reader.Execute()
        return dicom_image

    def save_to_folder(self, path: str):
        for label, structure in self.structures.items():
            sitk.WriteImage(structure, os.path.join(path, label + ".nii.gz"))

    def load_folder(self, path: str):
        for fol, subs, files in os.walk(path, followlinks=True):
            for file in files:
                if file.endswith(".nii.gz"):
                    self.structures[file.replace(".nii.gz", "")] = sitk.ReadImage(os.path.join(fol, file))

    def load_dicom_rtstruct(self,
                            rtstruct_file: str,
                            xy_scaling_factor: int = 1,
                            crop_masks: bool = False):

        contours = parser.parse_dcmrtstruct(rtstruct_file)
        for contour in contours:
            mask = converter.convert(contour=contour,
                                     dicom_image=self.reference_image,
                                     xy_scaling_factor=xy_scaling_factor,
                                     crop_masks=crop_masks)
            self.structures[contour.name] = mask

    def implode(self, label_map: Dict[str, int] | None = None):
        """
        :param label_map: Mapping of structures to implode. Key str must be exact match with the key in self.structures.
         Be careful that overlapping structures will be overwritten
        zby higher label integer. If not set, all structures are imploded in a non-deterministic order (again be careful)
        :return: sitk.Image
        """
        merged_arr = None
        ref_mask = None
        counter = 0
        for label, structure in self.structures.items():
            counter += 1
            # Instantiate merge_array
            if merged_arr is None:
                arr = sitk.GetArrayFromImage(structure)
                merged_arr = np.zeros_like(arr)

            if ref_mask is None:
                ref_mask = structure

            #
            if label_map:
                if label in label_map.keys():
                    arr = sitk.GetArrayFromImage(structure)
                    merged_arr[arr.astype(bool)] = label_map[label]
            else:
                arr = sitk.GetArrayFromImage(structure)
                merged_arr[arr.astype(bool)] = counter

        img = sitk.GetImageFromArray(merged_arr)
        img.CopyInformation(ref_mask)

        return img