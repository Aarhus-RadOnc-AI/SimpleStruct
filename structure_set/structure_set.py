import os
from typing import Dict, Union, List

import SimpleITK as sitk
import numpy as np

from structure_set.dicom import parser, converter, misc


class StructureSet:
    def __init__(self, reference_image: sitk.Image):
        self.reference_image: sitk.Image = reference_image

        self._rtstruct_contours: Union[Dict[str, Union[parser.Contour, str]]] = {}
        self._xy_scaling_factor: int = 1
        self._crop_masks: bool = False

        self._structures: Dict[str, Union[sitk.Image, None]] = {}

    def list_structure_names(self):
        return list(self._structures.keys())

    def get_structure(self, name):
        if not self._structures[name]:
            rtstruct = self._rtstruct_contours[name]
            if isinstance(rtstruct, str):
                self._structures[name] = sitk.ReadImage(rtstruct)
            else:
                self._structures[name] = converter.convert(contour=rtstruct,
                                                           xy_scaling_factor=self._xy_scaling_factor,
                                                           crop_masks=self._crop_masks,
                                                           dicom_image=self.reference_image)
        return self._structures[name]

    @staticmethod
    def parse_dicom_image(path):
        return misc.parse_dicom_image(path)

    def to_folder(self, path: str):
        for label in self.list_structure_names():
            sitk.WriteImage(self.get_structure(label), os.path.join(path, label + ".nii.gz"))

    def from_folder(self, path: str):
        for fol, subs, files in os.walk(path, followlinks=True):
            for file in files:
                if file.endswith(".nii.gz"):
                    self._rtstruct_contours[file.replace(".nii.gz", "")] = os.path.join(fol, file)

    def from_dicom_rtstruct(self,
                            rtstruct_file: str,
                            xy_scaling_factor: int = 1,
                            crop_masks: bool = False):
        try:
            for contour in parser.parse_dcmrtstruct(rtstruct_file):
                self._rtstruct_contours[contour.name] = contour
                self._structures[contour.name] = None

            self._xy_scaling_factor = xy_scaling_factor
            self._crop_masks = crop_masks
        except Exception as e:
            raise e

    def implode(self, labels: List[str] | None = None):
        """
        :param label_map: labels to implode. First item in the list will get integer=1, second=2 etc.
         Be careful that overlapping structures will be overwritten
        If not set, all structures are imploded in a non-deterministic order (again be careful)
        :return: sitk.Image
        """
        merged_arr = None
        ref_mask = None

        if not labels:
            labels = self.list_structure_names()

        for i, label in enumerate(labels):
            i += 1
            # Instantiate merge_array
            if merged_arr is None:
                arr = sitk.GetArrayFromImage(self.get_structure(label))
                merged_arr = np.zeros_like(arr)

            if ref_mask is None:
                ref_mask = self.get_structure(label)

            # Merge
            arr = sitk.GetArrayFromImage(self.get_structure(label))
            merged_arr[arr.astype(bool)] = i

        img = sitk.GetImageFromArray(merged_arr)
        img.CopyInformation(ref_mask)

        return img