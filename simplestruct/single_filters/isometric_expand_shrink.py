import SimpleITK as sitk
import numba as nb
import numpy as np
from numba import njit

from simplestruct.metrics.distance.hd import find_distance_for_coord
from simplestruct.single_filters import get_edge_of_structure


def get_spacing_as_np_array(structure: sitk.Image):
    return np.array(list(reversed(structure.GetSpacing())))

@njit
def _yield_new_coords(coords):
    new_coords = []
    for c in coords:
        for new_z in range(c[0] - 1, c[0] + 2):
            for new_y in range(c[1] - 1, c[1] + 2):
                for new_x in range(c[2] - 1, c[2] + 2):
                    new_coords.append((new_z, new_y, new_x))
    return nb.typed.List(new_coords)


@njit
def _expand_volume(structure: np.ndarray, ref_coords, coords, spacing, mm_radius):
    if len(coords) == 0:
        return

    new_coords = set()

    for coord in coords:
        if structure[coord] == 0:
            dist = find_distance_for_coord(other_coords=ref_coords, coord=coord, spacing_array=spacing)
            if dist <= mm_radius:
                structure[coord] = 1
                new_coords.add(coord)

    _expand_volume(structure=structure,
                   ref_coords=ref_coords,
                   coords=_yield_new_coords(new_coords),
                   spacing=spacing,
                   mm_radius=mm_radius)

    return structure


def isometric_expand_3d(structure: sitk.Image, mm_radius: float):
    spacing = get_spacing_as_np_array(structure)
    struct_arr = sitk.GetArrayFromImage(structure)
    ref_coords = np.argwhere(struct_arr)
    edge_coords = np.argwhere(get_edge_of_structure(struct_arr))
    new_coords = _yield_new_coords(edge_coords)
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
    sitk.WriteImage(ex, path.replace("HNCDL_003", "expand_test"))
