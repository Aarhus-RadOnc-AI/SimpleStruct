from typing import Tuple

import numpy as np

from simplestruct.metrics.hd import HD
import SimpleITK as sitk


class SurfaceDice:
    def __init__(self,
                 reference_image: sitk.Image,
                 other_image: sitk.Image,
                 ):
        self.hd = HD(reference_image=reference_image, other_image=other_image)

    def get_surface_dice(self,
                         tolerance: float = 1):
        distances = self.hd.get_distances(undirected=False)
        under_tolerance = distances <= tolerance
        surface_dice = np.count_nonzero(under_tolerance) / distances.shape[0]
        return surface_dice