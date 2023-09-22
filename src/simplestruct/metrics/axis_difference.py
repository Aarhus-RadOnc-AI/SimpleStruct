from typing import Tuple, Union

import SimpleITK as sitk
import numpy as np

from simplestruct.utils.type_functions import is_image, load_as_np_array


class AxisDiff:
    def __init__(self, reference_structure: Union[sitk.Image, np.ndarray], other_structure: [sitk.Image, np.ndarray]):
        self.reference_structure = load_as_np_array(reference_structure)
        self.other_structure = load_as_np_array(other_structure)

        self.diffs = {}
    def execute(self):

        for i in range(len(self.reference_structure.shape)):
            ax_ref = np.any(self.reference_structure, axis=i)
            ax_other = np.any(self.other_structure, axis=i)

            ax_ref_min, ax_ref_max = np.where(ax_ref)[0][[0, -1]]
            ax_other_min, ax_other_max = np.where(ax_other)[0][[0, -1]]

            self.diffs[i] = {"min": ax_other_min - ax_ref_min,
                             "max": ax_other_max - ax_ref_max}

    def get_axis_difference(self, axis = None):
        if len(self.diffs) == 0:
            self.execute()
        if not axis:
            return self.diffs
        else:
            if axis in self.diffs.keys():
                return self.diffs[axis]
            else:
                print(f"Invalid axis. Try one of {self.diffs.keys()}")
