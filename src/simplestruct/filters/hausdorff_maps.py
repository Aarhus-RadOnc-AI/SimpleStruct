from simplestruct.metrics.hd import HD
import SimpleITK as sitk
from typing import List
import numpy as np

class HausdorffMap:
    def __init__(self, reference_structure: sitk.Image, other_contours: List[sitk.Image]):
        self.reference_structure = reference_structure
        self.other_contours = other_contours
        self.hausdorff_map = None

    def execute(self):
        """
        Returns a np.ndarray, where columns are Z, Y, X coordinates in index 0:3 and 3: is the HD for the given coordinate
        for all "other_contours".
        Assumes that the contour integers are 1, so binarize your images beforehand.
        """
        arr = None
        for other_contour in self.other_contours:
            hd = HD(reference_image=self.reference_structure, other_image=other_contour, label_int=(1, 1))
            if arr is None:
                arr = hd.get_distance_matrix_ref_to_other()
            else:
                arr = np.insert(arr, -1, hd.get_distance_matrix_ref_to_other()[-1, :], axis=0)

        self.hausdorff_map = arr

def generate_hausdorff_map(reference_structure: sitk.Image, other_contours: List[sitk.Image]) -> HausdorffMap:
    hd_map = HausdorffMap(reference_structure=reference_structure, other_contours=other_contours)
    hd_map.execute()
    return hd_map