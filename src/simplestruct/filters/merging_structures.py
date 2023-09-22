import logging
from typing import List, Union

import SimpleITK as sitk
import numpy as np

from simplestruct.utils.type_functions import is_image



def _median_threshold(structure: np.ndarray, count: int) -> np.ndarray:
    return np.where(structure >= (count / 2), 1, 0)


def generate_median_structure(structures: Union[List[sitk.Image], List[np.ndarray]]) -> Union[sitk.Image, np.ndarray]:
    # Holds sum of structures
    ref = structures[0]

    sum_arr = np.zeros_like(sitk.GetArrayFromImage(structures[0])) if is_image(ref) else np.zeros_like(structures[0], np.uint8)

    for struct in structures:
        if is_image(ref):
            struct = sitk.GetArrayFromImage(struct)

        sum_arr += struct.astype(bool).astype(np.uint8)

    median_arr = _median_threshold(sum_arr, len(structures))

    if is_image(ref):
        median_img = sitk.GetImageFromArray(median_arr)
        median_img.CopyInformation(ref)
        return median_img
    else:
        return median_arr


def generate_median_structure_from_paths(structures: List[str]) -> sitk.Image:
    # Holds sum of structures
    ref = sitk.ReadImage(structures[0], sitk.sitkUInt8)

    sum_arr = np.zeros_like(sitk.GetArrayFromImage(ref))

    for path in structures:
        struct = sitk.GetArrayFromImage(sitk.ReadImage(path, sitk.sitkUInt8))
        sum_arr += struct.astype(bool).astype(np.uint8)
        del struct

    median_arr = _median_threshold(sum_arr, len(structures))

    median_img = sitk.GetImageFromArray(median_arr)
    median_img.CopyInformation(ref)
    return median_img
