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

        self.hausdorff_map = None
        for other_contour in self.other_contours:
            hd = HD(reference_image=self.reference_structure, other_image=other_contour)
            if self.hausdorff_map is None:
                self.hausdorff_map = hd.get_distance_matrix_ref_to_other()
            else:
                self.hausdorff_map = np.insert(self.hausdorff_map, -1, hd.get_distance_matrix_ref_to_other()[:, -1], axis=1)

    def get_hausdorff_map(self):
        if self.hausdorff_map is None:
            self.execute()
        return self.hausdorff_map


