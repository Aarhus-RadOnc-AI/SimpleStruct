import copy
from typing import List, Dict, Tuple

import SimpleITK as sitk
import numpy as np


def get_shape_of_structure(structure: sitk.Image, label_int: int = 1) -> Tuple[int]:
    f = sitk.LabelShapeStatisticsImageFilter()
    f.Execute(structure)
    bb_raw = f.GetBoundingBox(label=label_int)
    bb = np.zeros([3, ])
    bb[0] = bb_raw[0] + bb_raw[3]  # x max
    bb[1] = bb_raw[1] + bb_raw[4]  # y max
    bb[2] = bb_raw[2] + bb_raw[5]  # z max
    return tuple([int(b) for b in bb])


def get_extreme_origin_and_size_of_structures(structures: List[sitk.Image], label_int: int = 1):
    min_origin = None
    max_coord = None
    first = None
    for structure in structures:
        if first is None:
            first = copy.copy(structure)

        # For Origin
        origin = np.array(structure.GetOrigin())
        if min_origin is None:
            min_origin = origin
        else:
            origin_overlay = min_origin > origin
            min_origin[origin_overlay] = origin[origin_overlay]

        # For Max coord
        bb = get_shape_of_structure(structure=structure, label_int=label_int)
        last_coord = np.array(structure.TransformIndexToPhysicalPoint(bb))

        if max_coord is None:
            max_coord = last_coord
        else:
            max_coord_overlay = last_coord > max_coord
            max_coord[max_coord_overlay] = last_coord[max_coord_overlay]

    shape = np.array(first.TransformPhysicalPointToIndex(max_coord)) - np.array(
        first.TransformPhysicalPointToIndex(min_origin))
    shape += 1
    return tuple([float(o) for o in min_origin]), tuple([int(s) for s in shape])


def resample_structure_to_origin_and_shape(contour: sitk.Image, origin: Tuple, shape: Tuple):
    res_img = sitk.Resample(contour,
                            shape,
                            sitk.Transform(),
                            sitk.sitkNearestNeighbor,
                            origin,
                            contour.GetSpacing(),
                            contour.GetDirection(),
                            0,
                            contour.GetPixelIDValue(),
                            False)
    return res_img


def resample_structures_to_common_origin(structures: Dict[str, sitk.Image], label_int=1) -> Dict[str, sitk.Image]:
    """
    Resamples all structures to the bounding box of all label_int's in the set of structures. This is useful when
    you need to run metrics etc. across the sitk.Images.
    :param structures: Dict of name, structure.
    :param label_int: The integer, which
    :return: Dict of name, resampled_structure
    """
    # Find the max bounding box needed to encompass all contours in ss
    origin, shape = get_extreme_origin_and_size_of_structures(list(structures.values()), label_int=label_int)

    resampled_structures = {}  # Container for resampled structures
    for name, structure in structures.items():
        resampled_structures[name] = resample_structure_to_origin_and_shape(contour=structure, origin=origin, shape=shape)

    return resampled_structures
