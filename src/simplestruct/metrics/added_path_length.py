from typing import Union

import numpy as np
from simplestruct.filters import generate_edge_of_structure
import SimpleITK as sitk

from simplestruct.utils.type_checker import is_image


def apl(reference_structure: Union[sitk.Image, np.ndarray], other_structure: Union[sitk.Image, np.ndarray], normalized=True):
    """
    It must be possible to cast contour images as boolean arrays - only 0 and one other label should be present.
    """
    if is_image(reference_structure):
        reference_structure = sitk.GetArrayFromImage(reference_structure)

    if is_image(other_structure):
        other_structure = sitk.GetArrayFromImage(other_structure)

    gt_edge = generate_edge_of_structure(reference_structure)
    other_edge = generate_edge_of_structure(other_structure)

    ## Edge case if prediction is all false and should not be. If so, return full size of prediction
    if np.count_nonzero(gt_edge) == 0:
        apl = np.count_nonzero(other_edge)
    else:
        apl = (gt_edge < other_edge).astype(int).sum()

    if normalized:
        apl = apl / np.count_nonzero(other_edge)

    return apl
