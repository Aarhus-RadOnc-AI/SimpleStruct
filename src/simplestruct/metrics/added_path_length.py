from typing import Union

import numpy as np
from simplestruct.filters import generate_edge_of_structure
import SimpleITK as sitk

from simplestruct.utils.type_checker import is_image

class APL:

    def __init__(self, reference_structure: Union[sitk.Image, np.ndarray], other_structure: Union[sitk.Image, np.ndarray]):
        """
        It must be possible to cast contour images as boolean arrays - only 0 and one other label should be present.
        """
        if is_image(reference_structure):
            self.reference_structure = sitk.GetArrayFromImage(reference_structure)
        else:
            self.reference_structure = reference_structure

        if is_image(other_structure):
            self.other_structure = sitk.GetArrayFromImage(other_structure)
        else:
            self.other_structure = other_structure

        self.gt_edge = None
        self.other_edge = None
    def execute(self):
        self.gt_edge = generate_edge_of_structure(self.reference_structure)
        self.other_edge = generate_edge_of_structure(self.other_structure)

        ## Edge case if prediction is all false and should not be. If so, return full size of prediction
        if np.count_nonzero(self.gt_edge) == 0:
            self.raw_apl = np.count_nonzero(self.other_edge)
        else:
            self.raw_apl = (self.gt_edge < self.other_edge).astype(int).sum()

        self.norm_apl = self.raw_apl / np.count_nonzero(self.other_edge)

    def get_apl(self, normalized=True):
        if normalized:
            return self.norm_apl
        else:
            return self.raw_apl
