import SimpleITK as sitk
import numpy as np
from numba import njit, prange

from simplestruct.metrics.hd import find_distance_for_coord

from simplestruct.filters import generate_edge_of_structure
from simplestruct.utils.njit_wrapper import njit_if_loaded


def get_spacing_as_np_array(structure: sitk.Image):
    return np.array(list(reversed(structure.GetSpacing())))


@njit_if_loaded
def _yield_new_coords(coords: np.ndarray, mm_radius: float):
    """
    Yields only if coord dist is below mm_radius
    """
    new_coords = set()
    # new_coords = np.empty(coords.shape[0])

    for i in prange(coords.shape[0]):
        coord = np.array([int(c) for c in coords[i, 1:]])

        if coords[i, 0] <= mm_radius:
            for new_z in range(coord[0] - 1, coord[0] + 2):
                for new_y in range(coord[1] - 1, coord[1] + 2):
                    for new_x in range(coord[2] - 1, coord[2] + 2):
                        new_coords.add((new_z, new_y, new_x))
    l = list(new_coords)
    if len(l) == 0:
        l = [(-1, -1, -1)]
    arr = np.array(l)
    return arr


@njit_if_loaded(parallel=True)
def _expand_volume(structure: np.ndarray, ref_coords, coords, spacing, mm_radius):
    dist_coords = np.empty((len(coords), 4))
    dist_coords[:, 1:] = coords
    for i in prange(dist_coords.shape[0]):
        coord = np.array([int(c) for c in dist_coords[i, 1:]])

        if structure[coord[0], coord[1], coord[2]] == 0:
            dist = find_distance_for_coord(other_coords=ref_coords, coord=coord, spacing_array=spacing)
            dist_coords[i, 0] = dist
        else:
            dist_coords[i, 0] = mm_radius + 1

    # Set in structure array to 1 where dist is ge to mm_radius
    for i in prange(dist_coords.shape[0]):
        if dist_coords[i, 0] <= mm_radius:
            coord = np.array([int(c) for c in dist_coords[i, 1:]])
            structure[coord[0], coord[1], coord[2]] = 1

    new_coords = _yield_new_coords(dist_coords, mm_radius)
    if new_coords.shape[0] == 1:
        if new_coords[0, 0] == -1:
            return
    else:
        _expand_volume(structure=structure,
                          ref_coords=ref_coords,
                          coords=new_coords,
                          spacing=spacing,
                          mm_radius=mm_radius)

def isometric_expand_3d(structure: sitk.Image, mm_radius: float):
    spacing = get_spacing_as_np_array(structure)
    struct_arr = sitk.GetArrayFromImage(structure)
    ref_coords = np.argwhere(struct_arr)
    edge_coords = np.argwhere(generate_edge_of_structure(struct_arr))
    edge_coords = np.insert(edge_coords, 0, 0, axis=1)  # Mock distance
    new_coords = _yield_new_coords(edge_coords, 10)  # Mock margin to make sure new_coords are generated
    _expand_volume(structure=struct_arr,
                   ref_coords=ref_coords,
                   coords=new_coords,
                   spacing=spacing,
                   mm_radius=mm_radius)

    expanded = sitk.GetImageFromArray(struct_arr)
    expanded.CopyInformation(structure)
    return expanded


if __name__ == "__main__":
    path = "/home/mathis/work/studies/3_GTV/documentation/end_to_end/eb552726dace/labelsTs/HNCDL_003.nii.gz"
    struct = sitk.ReadImage(path) == 1
    ex = isometric_expand_3d(struct, 10)
    print(ex)
    sitk.WriteImage(ex, path.replace("HNCDL_003", "expand_test"))
