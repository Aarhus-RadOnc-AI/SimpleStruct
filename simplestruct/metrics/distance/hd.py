import logging

import SimpleITK as sitk
import numpy as np
from numba import njit

from simplestruct.single_filters.get_edge_of_structure import get_edge_of_structure


def find_distance_for_coord(coord, other_coords, spacing_array):
    vectors = other_coords - coord
    vectors = np.multiply(vectors, spacing_array)
    vectors = np.power(vectors, 2)
    vectors = np.sum(vectors, axis=1)
    vector_lengths = np.sqrt(vectors)
    coord_hd = np.min(vector_lengths)
    return coord_hd


class HD:
    def __init__(self, reference_image: sitk.Image, other_image: sitk.Image, label_int=(1, 1)):
        self.ref_img = reference_image
        self.other_img = other_image
        self.ref_arr = get_edge_of_structure(sitk.GetArrayFromImage(self.ref_img), label_int[0])
        self.other_arr = get_edge_of_structure(sitk.GetArrayFromImage(self.other_img), label_int[1])
        self.spacing_arr = np.array(self.ref_img.GetSpacing())[-1::-1]

        self.distance_matrix_ref_to_other = None
        self.distance_matrix_other_to_ref = None

    def _calculate_distance_matrix(self, reference: np.ndarray, other: np.ndarray) -> np.ndarray:
        """
        This function gives the directed distance from reference to other contour.
        :return:
        """
        ref_coords = np.argwhere(reference)
        other_coords = np.argwhere(other)
        print(ref_coords.shape)
        distance_matrix = np.empty((4, ref_coords.shape[0]))  # columns are Z, Y, X, hausdorff distance for this point
        distance_matrix[:3, :] = ref_coords.T
        for i, coord in enumerate(ref_coords):
            distance_matrix[3, i] = find_distance_for_coord(coord=coord,
                                                            other_coords=other_coords,
                                                            spacing_array=self.spacing_arr)

        return distance_matrix

    def _generate_distance_matrices(self, undirected=True):
        if self.distance_matrix_ref_to_other is None:
            self.distance_matrix_ref_to_other = self._calculate_distance_matrix(self.ref_arr, self.other_arr)
        if undirected and self.distance_matrix_other_to_ref is None:
            self.distance_matrix_other_to_ref = self._calculate_distance_matrix(self.other_arr, self.ref_arr)

    def get_distances(self, undirected=True):
        self._generate_distance_matrices(undirected=undirected)
        if undirected:
            return np.concatenate([self.distance_matrix_ref_to_other[3, :], self.distance_matrix_other_to_ref[3, :]],
                                  axis=1)
        else:
            return self.distance_matrix_ref_to_other[3, :]

    def func_on_min_distances(self, func=np.max, undirected=True):
        return func(self.get_distances(undirected))

    def get_max_min_hd(self, undirected=True):
        return self.func_on_min_distances(func=np.max, undirected=undirected)

    def get_avg_min_hd(self, undirected=True):
        return self.func_on_min_distances(func=np.mean, undirected=undirected)

    def get_percentile_min_hd(self, percentile=0.95, undirected=True):
        arr = self.func_on_min_distances(np.sort, undirected=undirected)
        return arr[int(arr.shape[0] * percentile)]
