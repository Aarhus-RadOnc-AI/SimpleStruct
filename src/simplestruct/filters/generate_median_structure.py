from typing import List, Union

import SimpleITK as sitk
import numpy as np

from simplestruct.utils.type_functions import is_image


def _sum_arrays(structures: List[np.ndarray]) -> np.ndarray:
    sum_arr = np.zeros(list(reversed(structures[0].GetSize())))

    for s in structures:
        arr = s
        sum_arr += arr


def _median_threshold(structure: np.ndarray, count: int) -> np.ndarray:
    return np.where(structure >= (count / 2))


def generate_median_structures(structures: Union[List[sitk.Image], List[np.ndarray]]) -> Union[sitk.Image, np.ndarray]:
    # Holds sum of structures
    ref = structures[0]
    if is_image(ref):
        structures = [sitk.GetArrayFromImage(s) for s in structures]

    summed_arr = _sum_arrays(structures)
    median_arr = _median_threshold(summed_arr, len(structures))

    if is_image(ref):
        median_img = sitk.GetImageFromArray(median_arr)
        median_img.CopyInformation(ref)
        return median_img
    else:
        return median_arr
