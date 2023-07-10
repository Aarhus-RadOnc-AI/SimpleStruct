import os.path
from typing import Union, List, Dict, Any

import SimpleITK as sitk
import numpy as np
import pydantic
from dcmrtstruct2nii.adapters.convert.rtstructcontour2mask import DcmPatientCoords2Mask
from dcmrtstruct2nii.adapters.input.contours.rtstructinputadapter import RtStructInputAdapter
from pydantic import ConfigDict


class Sequence(pydantic.BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: str
    points: Dict[str, List[float]]

    @property
    def vertices(self):
        return np.column_stack([self.points["x"],
                               self.points["y"],
                               self.points["z"]])


class Structure(pydantic.BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    roi_number: int
    referenced_frame: str
    display_color: Any
    sequences: List[Sequence] = []

    structure_set: pydantic.BaseModel
    _raster: Union[sitk.Image, None] = None
    raster_scaling_factor: int
    raster_crop: bool
    _updated: bool = False

    @property
    def vertices(self):
        seqs = []
        for i, seq in enumerate(self.sequences):
            seqs.append(np.insert(seq.vertices, 0, i, axis=1))

        return np.concatenate(seqs)

    @property
    def raster(self):
        if self._raster is None:
            self._raster = DcmPatientCoords2Mask().convert(rtstruct_contours=self.model_dump()["sequences"],
                                                          dicom_image=self.structure_set.image,
                                                          crop_mask=self.raster_crop,
                                                          xy_scaling_factor=self.raster_scaling_factor)
        return self._raster


class StructureSet(pydantic.BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rtstruct_file: str
    image_path: Union[str, None] = None
    _image: Union[sitk.Image, None] = None
    structures: Dict[str, Structure] = {}
    raster_scaling_factor: Union[int, Dict[str, int]] = 1  # If integer, apply on all. If Dict, structure_name: factor is used
    raster_crop: Union[bool, Dict[str, bool]] = False  # If integer, apply on all. If Dict, structure_name: bool is used

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._load_rtstruct()

    @property
    def image(self):
        if not self._image:
            if os.path.isdir(self.image_path):
                dicom_reader = sitk.ImageSeriesReader()
                dicom_file_names = dicom_reader.GetGDCMSeriesFileNames(self.image_path)
                dicom_reader.SetFileNames(dicom_file_names)
                self._image = dicom_reader.Execute()
            elif os.path.isfile(self.image_path):
                self._image = sitk.ReadImage(self.image_path)
            else:
                raise Exception(
                    "Image path must be either path to dicom image folder or path to a SimpleITK-loadable image")
        return self._image

    def _load_rtstruct(self):
        rtstructs = RtStructInputAdapter().ingest(input_file=self.rtstruct_file)
        for rtstruct in rtstructs:
            # Fill empty spaces
            if not rtstruct["sequence"]:
                rtstruct["sequence"] = []
            if rtstruct["name"] not in self.structures.keys():
                struct = Structure(name=rtstruct["name"],
                                   roi_number=rtstruct["roi_number"],
                                   referenced_frame=rtstruct["referenced_frame"],
                                   display_color=rtstruct["display_color"],
                                   sequences=[Sequence(**s) for s in rtstruct["sequence"]],
                                   structure_set=self,
                                   raster_scaling_factor=self.raster_scaling_factor if isinstance(self.raster_scaling_factor, int) else self.raster_scaling_factor[rtstruct["name"]],
                                   raster_crop=self.raster_crop if isinstance(self.raster_crop, bool) else self.raster_crop[rtstruct["name"]])

                self.structures[rtstruct["name"]] = struct
            else:
                self.structures[rtstruct["name"]]["sequences"] += [Sequence(**s) for s in rtstruct["sequence"]]

    def get_structure_names(self):
        return list(self.structures.keys())

    def get_structure(self, name: str):
        return self.structures[name]

if __name__ == "__main__":
    ss = StructureSet(
        rtstruct_file="/home/mathis/Documents/Studies/3_GTV/documentation/end_to_end/eb552726dace/pipeline/HNCDL_003/out/segmentation.dcm",
        image_path="/home/mathis/Documents/Studies/3_GTV/documentation/end_to_end/eb552726dace/pipeline/HNCDL_003/in/1.2.840.10008.5.1.4.1.1.2/1.2.826.0.1.3680043.2.135.737080.33450775.7.1548105591.45.37/",
        raster_crop=True,
        raster_scaling_factor=10)

    print(ss.get_structure("AI_GTVt").vertices)

