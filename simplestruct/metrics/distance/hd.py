import SimpleITK as sitk
import numpy as np

from simplestruct.single_filters.get_edge_of_structure import get_edge_of_structure

class HD:
    def __init__(self, reference_image: sitk.Image, other_image: sitk.Image, label_int=(1, 1)):
        self.ref_img = reference_image
        self.other_img = other_image
        self.ref_arr = get_edge_of_structure(sitk.GetArrayFromImage(self.ref_img), label_int[0])
        self.other_arr = get_edge_of_structure(sitk.GetArrayFromImage(self.other_img), label_int[1])
        self.spacing_arr = np.array(self.ref_img.GetSpacing()[-1:])
        self.distance_matrix = None

    def set_distance_matrix(self):
        ref_coords = np.argwhere(self.ref_arr)
        other_coords = np.argwhere(self.other_arr)


        ref = np.empty((ref_coords.shape[0], other_coords.shape[0]))
        for i, row in enumerate(ref_coords):
            ref[i,:] = np.sqrt(np.sum(np.power(np.multiply((other_coords - row), self.spacing_arr), 2), axis=1))

    def max_min_hd(self):
        return self.distance_matrix.min(axis=1).max()

