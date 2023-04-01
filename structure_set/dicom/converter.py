import logging
from typing import Tuple

import SimpleITK as sitk
import numpy as np
from numba import njit
from skimage import draw

from structure_set.dicom import parser, misc


def _get_transform_matrix(spacing: Tuple, direction: Tuple):
    """
    this returns the basics needed to run _transform_physical_point_to_continuous_index
    """
    s = np.array(spacing)
    d = np.array(direction).reshape(3, 3)
    m_IndexToPhysicalPoint = np.multiply(d, s)
    m_PhysicalPointToIndex = np.linalg.inv(m_IndexToPhysicalPoint)

    return m_PhysicalPointToIndex


def xor_update_np_mask(np_mask, filled_poly, z):
    overlay = np.logical_xor(np_mask[z, :, :], filled_poly)
    np_mask[z, :, :] = overlay


@njit
def _transform_physical_point_to_continuous_index(coords, m_PhysicalPointToIndex, origin):
    """
    This method does the same as SimpleITK's TransformPhysicalPointToContinuousIndex, but in a vectorized fashion.
    The implementation is based on ITK's code found in https://itk.org/Doxygen/html/itkImageBase_8h_source.html#l00497 and
    https://discourse.itk.org/t/solved-transformindextophysicalpoint-manually/1031/2
    """

    if m_PhysicalPointToIndex is None:
        raise Exception("Run set transform variables first!")

    pts = np.empty_like(coords)
    pts[:, 0] = coords[:, 0]  # Index of contour
    pts[:, 1] = coords[:, 1] - origin[0]  # x
    pts[:, 2] = coords[:, 2] - origin[1]  # y
    pts[:, 3] = coords[:, 3] - origin[2]  # z

    pts[:, 1:] = pts[:, 1:].copy() @ m_PhysicalPointToIndex

    return pts


def get_cropped_origin(stacked_coords):
    float_min = np.min(stacked_coords[:, 1:], axis=0)
    return float_min


def stack_coords(contour: parser.Contour) -> np.ndarray:
    coords = None
    for i, sequence in enumerate(contour.sequences):
        if sequence.type.upper() not in ['CLOSED_PLANAR', 'INTERPOLATED_PLANAR']:
            logging.info(f'Skipping unnamed contour, unsupported type: {sequence.type}')
            continue

        # Stack coordinate components to one array
        temp_coords = sequence.points
        stack = np.column_stack((
            [i for u in range(len(sequence.points.x))],
            sequence.points.x,
            sequence.points.y,
            sequence.points.z)
        )  # Stack column 0 is index of contour, then x, y, z.

        if coords is None:
            coords = stack
        else:
            coords = np.concatenate([coords, stack])

    return coords


def get_shape(idx_pts):
    maxs = np.ceil(np.max(idx_pts[:, 1:], axis=0)).astype(int) + 1

    return maxs

def convert(contour: parser.Contour,
            xy_scaling_factor: int,
            crop_masks: bool,
            dicom_image: sitk.Image) -> sitk.Image:

    spacing = misc.scale_information_tuple(information_tuple=dicom_image.GetSpacing(),
                                            xy_scaling_factor=xy_scaling_factor,
                                            up=False,
                                            out_type=float)
    m_PhysicalPointToIndex = _get_transform_matrix(spacing=spacing,
                                                   direction=dicom_image.GetDirection())

    # Arrange contours into an array shape (n, 4), where column order is contour_index, x, y, z
    stacked_coords = stack_coords(contour=contour)

    # Origin set to minimum of x, y and z
    origin = get_cropped_origin(stacked_coords)

    # Index of coords
    idx_pts = _transform_physical_point_to_continuous_index(stacked_coords,
                                                            m_PhysicalPointToIndex=m_PhysicalPointToIndex,
                                                            origin=origin)
    # Get Shape for rastering
    shape = get_shape(idx_pts)

    np_mask = np.zeros(list(reversed(shape)), dtype=np.uint8)
    for idx in np.unique(idx_pts[:, 0]):
        pts = idx_pts[idx_pts[:, 0] == idx][:, 1:]  # Slice to only get coordinates
        z = int(pts[0, 2])  # Get z of the index

        try:
            # Draw the polygon and xor update np_mask
            filled_poly = draw.polygon2mask((shape[1], shape[0]), pts[:, 1::-1])
            xor_update_np_mask(np_mask=np_mask, filled_poly=filled_poly, z=z)
        except Exception as e:
            raise e

    # np_mask to image
    mask = sitk.GetImageFromArray(np_mask.astype(np.uint8))  # Had trouble with the type. Use np.uint8!

    # Set image meta
    arr = sitk.GetArrayFromImage(mask)
    mask.SetDirection(dicom_image.GetDirection())
    mask.SetOrigin(origin)
    mask.SetSpacing(spacing)

    # If not crop, then resample to align with the original dicom image.
    if not crop_masks:
        mask = sitk.Resample(mask,
                             misc.scale_information_tuple(dicom_image.GetSize(),
                                                           xy_scaling_factor=xy_scaling_factor,
                                                           out_type=int,
                                                           up=True),
                             sitk.Transform(),
                             sitk.sitkNearestNeighbor,
                             dicom_image.GetOrigin(),
                             spacing,
                             dicom_image.GetDirection())
    return mask
