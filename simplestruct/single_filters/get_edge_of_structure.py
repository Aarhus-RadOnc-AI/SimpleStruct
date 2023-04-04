import numpy as np
from numba import njit

@njit
def get_edge_of_structure(structure: np.ndarray, label_int: int = 1) -> np.ndarray:
    """
    Mask must only contain 0 for background and 1 for mask. Array must be ordered [z, y, x]
    :param mask:
    :return:
    """

    mask = (structure == label_int)

    edge = np.zeros_like(mask)
    for z in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
            for x in range(0, mask.shape[2]):
                sum = np.sum(mask[z, y - 1:y + 2, x - 1:x + 2])
                if sum < 9:
                    edge[z, y, x] = mask[z, y, x]
    return edge

