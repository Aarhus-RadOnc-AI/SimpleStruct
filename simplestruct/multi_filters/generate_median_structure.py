from typing import List

import SimpleITK as sitk
import numpy as np

def generate_median_structures(structures: List[sitk.Image]) -> sitk.Image:
    # Holds sum of structures
    sum_arr = np.zeros(list(reversed(structures[0].GetSize())))

    for s in structures:
        arr = sitk.GetArrayFromImage(s)
        sum_arr += arr

    median_arr = (sum_arr >= (len(structures) / 2)).astype(np.uint8)
    median_img = sitk.GetImageFromArray(median_arr)

    median_img.CopyInformation(structures[0])

    return median_img
