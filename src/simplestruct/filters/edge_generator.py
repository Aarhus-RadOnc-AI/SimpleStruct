import numpy as np

from simplestruct.utils.njit_wrapper import njit


@njit
def generate_edge_of_structure(structure: np.ndarray, use_2d: bool = True) -> np.ndarray:
    """
    Is binarized, so 0 is background and 1-n is contour. Array is ordered [z, y, x]
    :param mask:
    :return:
    """

    mask = structure != 0
    edge = np.zeros_like(mask)
    for z in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
            for x in range(0, mask.shape[2]):
                if use_2d:
                    i_sum = np.sum(mask[z, y - 1:y + 2, x - 1:x + 2])
                    if i_sum < 9:
                        edge[z, y, x] = mask[z, y, x]
                else:
                    i_sum = np.sum(mask[z - 1: z + 2, y - 1:y + 2, x - 1:x + 2])
                    if i_sum < 27:
                        edge[z, y, x] = mask[z, y, x]

    return edge

@njit
def count_edge_of_structure(structure: np.ndarray, use_3d: bool = False) -> int:
    edge = generate_edge_of_structure(structure, use_3d)
    return np.count_nonzero(edge)