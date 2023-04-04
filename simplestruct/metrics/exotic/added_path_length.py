import numpy as np
from simplestruct.single_filters.get_edge_of_structure import get_edge_of_structure
import SimpleITK as sitk

def calculate_added_path_length(reference_structure: sitk.Image, other_structure: sitk.Image):
    """
    It must be possible to cast contour images as boolean arrays - only 0 and one other label should be present.
    """

    gt_edge = get_edge_of_structure(sitk.GetArrayFromImage(reference_structure))
    pred_edge = get_edge_of_structure(sitk.GetArrayFromImage(other_structure))

    ## Edge case if prediction is all false and should not be. If so, return full size of prediction
    if np.count_nonzero(gt_edge) == 0:
        apl = np.count_nonzero(pred_edge)
    else:
        apl = (gt_edge > pred_edge).astype(int).sum()

    return apl
