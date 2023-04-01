import logging
import traceback
from typing import List, NewType
import pydicom
from pydicom.errors import InvalidDicomError
import pydantic
import numpy as np

"""
This function is adapted from dcmrtstruct2nii by Thomas Phil https://github.com/Sikerdebaard/dcmrtstruct2nii
"""


class Points(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


class Sequence(pydantic.BaseModel):
    type: str | None = None
    points: Points | None = None


class Contour(pydantic.BaseModel):
    name: str | None = None
    roi_number: str | None = None
    referenced_frame: str | None = None
    display_color: str | None = None
    sequences: List[Sequence] = []


Contours = NewType("Contours", List[Contour])


def parse_dcmrtstruct(dcmrtstruct_file, skip_contours=False) -> Contours:
    '''
        Load RT Struct DICOM from input_file and output intermediate format
        :param input_file: Path to the dicom rt-struct file
        :return: multidimensional array with ROI(s)
    '''

    try:
        rt_struct_image = pydicom.read_file(dcmrtstruct_file)

        if not hasattr(rt_struct_image, 'StructureSetROISequence'):
            raise InvalidDicomError()

    except Exception as e:
        logging.error(str(e))
        logging.error(traceback.format_exc())
        raise e

    # lets extract the ROI(s) and dcmrtstruct2nii it to an intermediate format
    contours = Contours([])

    # first create a map so that we can easily trace referenced_roi_number back to its metadata
    metadata_mappings = {}
    for contour_metadata in rt_struct_image.StructureSetROISequence:
        metadata_mappings[contour_metadata.ROINumber] = contour_metadata

    for contour_sequence in rt_struct_image.ROIContourSequence:
        contour_data = Contour()

        metadata = metadata_mappings[contour_sequence.ReferencedROINumber]  # retrieve metadata

        # I'm not sure if these attributes are always present in the metadata and contour_sequence
        # so I decided to write this in a defensive way.

        if hasattr(metadata, 'ROIName'):
            contour_data.name = metadata.ROIName

        if hasattr(metadata, 'ROINumber'):
            contour_data.roi_number = metadata.ROINumber

        if hasattr(metadata, 'ReferencedFrameOfReferenceUID'):
            contour_data.referenced_frame = metadata.ReferencedFrameOfReferenceUID

        if hasattr(contour_sequence, 'ROIDisplayColor') and len(contour_sequence.ROIDisplayColor) > 0:
            contour_data.display_color = contour_sequence.ROIDisplayColor

        if not skip_contours and hasattr(contour_sequence, 'ContourSequence') and len(
                contour_sequence.ContourSequence) > 0:
            for contour in contour_sequence.ContourSequence:
                points = Points(
                    x=np.array([contour.ContourData[index] for index in range(0, len(contour.ContourData), 3)] if hasattr(
                        contour, 'ContourData') else None),
                    y=np.array([contour.ContourData[index + 1] for index in range(0, len(contour.ContourData), 3)] if hasattr(
                        contour, 'ContourData') else None),
                    z=np.array([contour.ContourData[index + 2] for index in range(0, len(contour.ContourData), 3)] if hasattr(
                        contour, 'ContourData') else None)
                )
                sequence = Sequence(
                    type=(contour.ContourGeometricType if hasattr(contour, 'ContourGeometricType') else 'unknown'),
                    points=points
                )
                contour_data.sequences.append(sequence)
        if len(contour_data.sequences) != 0:
            # only add contour if we successfully extracted (some) data
            contours.append(contour_data)

    return contours
