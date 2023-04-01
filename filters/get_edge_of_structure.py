import numpy as np
from numba import njit
import SimpleITK as sitk

@njit
def get_edge_of_structure(structure: sitk.Image, label_int: int =1) -> sitk.Image:
    """
    Mask must only contain 0 for background and 1 for mask.
    :param mask:
    :return:
    """

    mask = sitk.GetArrayFromImage(structure)
    mask = (mask == label_int)

    edge = np.zeros_like(mask)
    for z in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
            for x in range(0, mask.shape[2]):
                sum = np.sum(mask[z, y - 1:y + 2, x - 1:x + 2])
                if sum < 9:
                    edge[z, y, x] = mask[z, y, x]
    edge = sitk.GetImageFromArray(edge)
    edge.CopyInformation(structure)
    return edge

