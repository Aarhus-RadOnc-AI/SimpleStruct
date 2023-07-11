import numpy as np
from numba import njit
import SimpleITK as sitk
from typing import Union

@njit
def get_edge_of_structure(structure: np.ndarray, label_int: int = 1, use_3d: bool = False) -> np.ndarray:
    """
    Mask must only contain 0 for background and 1 for mask. Array must be ordered [z, y, x]
    Input structure must be in np.ndarray, else njit won't work.
    :param mask:
    :return:
    """

    mask = (structure == label_int)
    edge = np.zeros_like(mask)
    for z in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
            for x in range(0, mask.shape[2]):
                if not use_3d:
                    i_sum = np.sum(mask[z, y - 1:y + 2, x - 1:x + 2])
                    if i_sum < 9:
                        edge[z, y, x] = mask[z, y, x]
                else:
                    i_sum = np.sum(mask[z - 1: z + 2, y - 1:y + 2, x - 1:x + 2])
                    if i_sum < 27:
                        edge[z, y, x] = mask[z, y, x]
    return edge

