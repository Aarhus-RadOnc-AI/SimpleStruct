from typing import Tuple

import SimpleITK as sitk
import numpy as np


def count_axis_difference(ref_image: sitk.Image, other_image: sitk.Image, axis=0) -> Tuple[int, int]:
    ref_arr = sitk.GetArrayFromImage(ref_image)
    other_arr = sitk.GetArrayFromImage(other_image)
    z_ref = np.any(ref_arr, axis=axis)
    z_other = np.any(other_arr, axis=axis)

    z_ref_min, z_ref_max = np.where(z_ref)[0][[0, -1]]
    z_other_min, z_other_max = np.where(z_other)[0][[0, -1]]

    return z_other_min - z_other_min, z_other_max - z_ref_max
