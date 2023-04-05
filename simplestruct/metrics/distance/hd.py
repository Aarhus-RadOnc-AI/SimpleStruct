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

        self.distance_matrix_directed = None
        self.distance_matrix_undirected = None

    def calculate_distance_matrix_directed(self):
        ref_coords = np.argwhere(self.ref_arr)
        other_coords = np.argwhere(self.other_arr)

        self.distance_matrix_directed = np.zeros((ref_coords.shape[0]))
        for i, coord in enumerate(other_coords):
            self.distance_matrix_directed[i] = find_distance_for_coord(coord=coord,
                                                                         other_coords=ref_coords,
                                                                         spacing_array=self.spacing_arr)

    def calculate_distance_matrix_undirected(self):
        ref_coords = np.argwhere(self.ref_arr)
        other_coords = np.argwhere(self.other_arr)

        self.distance_matrix_undirected = np.zeros((ref_coords.shape[0]))
        for i, coord in enumerate(ref_coords):
            self.distance_matrix_undirected[i] = find_distance_for_coord(coord=coord,
                                                                       other_coords=other_coords,
                                                                       spacing_array=self.spacing_arr)

    def calculate_distance_matrices(self, undirected=True):
        if self.distance_matrix_directed is None:
            self.calculate_distance_matrix_directed()
        if undirected and self.distance_matrix_undirected is None:
            self.calculate_distance_matrix_undirected()

    def max_min_hd(self, undirected=True):
        self.calculate_distance_matrices(undirected)
        if undirected:
            return np.max(np.array([np.max(self.distance_matrix_directed), np.max(self.distance_matrix_undirected)]))
        else:
            return np.max(self.distance_matrix_directed)
