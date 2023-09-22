import functools
from multiprocessing.pool import ThreadPool

from simplestruct.metrics.hd import HD
import SimpleITK as sitk
from typing import List
import numpy as np

class HausdorffMap:
    def __init__(self, reference_structure: sitk.Image, other_contours: List[sitk.Image], processes=8):
        self.reference_structure = reference_structure
        self.other_contours = other_contours
        self.processes = processes
        self.hausdorff_map = None

    def execute(self):
        """
        Returns a np.ndarray, where columns are Z, Y, X coordinates in index 0:3 and 3: is the HD for the given coordinate
        for all "other_contours".
        Assumes that the contour integers are 1, so binarize your images beforehand.
        """

        def process(ref, other):
            hd = HD(reference_image=ref, other_image=other)
            return hd.get_distance_matrix_ref_to_other()

        tp = ThreadPool(self.processes)
        maps = tp.starmap(process, [(self.reference_structure, other) for other in self.other_contours])
        tp.close()
        tp.join()

        self.hausdorff_map = None
        for map in maps:
            if self.hausdorff_map is None:
                self.hausdorff_map = map
            else:
                self.hausdorff_map = np.insert(self.hausdorff_map, -1, map[:, -1], axis=1)

    def get_hausdorff_map(self):
        if self.hausdorff_map is None:
            self.execute()
        return self.hausdorff_map

    def get_summarized_hausdorff_map(self, func=None):
        """
        Summarize hausdorff map with an arbitrary (numpy) function. Default is np.mean(arr, axis = 1).
        """
        if not func:
            func = functools.partial(np.mean, axis=1)
        arr = np.empty([self.hausdorff_map.shape[0], 4])
        arr[:, 3] = func(self.hausdorff_map[:, 3:])
        return arr

