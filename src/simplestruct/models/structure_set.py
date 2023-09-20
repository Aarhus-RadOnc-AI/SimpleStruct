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

    image: Union[sitk.Image, None] = None
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
    def raster_image(self):
        if self._raster is None:
            self._raster = DcmPatientCoords2Mask().convert(rtstruct_contours=self.model_dump()["sequences"],
                                                          dicom_image=self.image,
                                                          crop_mask=self.raster_crop,
                                                          xy_scaling_factor=self.raster_scaling_factor)
        return self._raster

    @property
    def raster_array(self):
        return sitk.GetArrayFromImage(self.raster_image)


class StructureSet(pydantic.BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rtstruct_file: str
    image_path: Union[str, None] = None
    _image: Union[sitk.Image, None] = None
    structures: Dict[str, Structure] = {}
    raster_scaling_factor: Union[int, Dict[str, int]] = 1  # If integer, apply on all. If Dict, structure_name: factor is used
    raster_crop: Union[bool, Dict[str, bool]] = False  # If integer, apply on all. If Dict, structure_name: bool is used

    @property
    def image(self):
        if not self.image_path:
            return
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


    def __init__(self, **data: Any):
        super().__init__(**data)
        self._load_rtstruct()


    def _load_rtstruct(self):
        rtstructs = RtStructInputAdapter().ingest(input_file=self.rtstruct_file)
        for rtstruct in rtstructs:
            # Fill empty spaces
            if not "sequence" in rtstruct.keys():
                rtstruct["sequence"] = []
            if rtstruct["name"] not in self.structures.keys():
                struct = Structure(name=rtstruct["name"],
                                   roi_number=rtstruct["roi_number"],
                                   referenced_frame=rtstruct["referenced_frame"],
                                   display_color=rtstruct["display_color"],
                                   sequences=[Sequence(**s) for s in rtstruct["sequence"]],
                                   image=self.image,
                                   raster_scaling_factor=self.raster_scaling_factor if isinstance(self.raster_scaling_factor, int) else self.raster_scaling_factor[rtstruct["name"]],
                                   raster_crop=self.raster_crop if isinstance(self.raster_crop, bool) else self.raster_crop[rtstruct["name"]])

                self.structures[rtstruct["name"]] = struct
            else:
                self.structures[rtstruct["name"]]["sequences"] += [Sequence(**s) for s in rtstruct["sequence"]]

    def get_structure_names(self):
        return list(self.structures.keys())

    def get_structure(self, name: str):
        return self.structures[name]

    def implode(self, labels: Union[List[str], None] = None):
        """
        :param label_map: labels to implode. First item in the list will get integer=1, second=2 etc.
         Be careful that overlapping structures will be overwritten
        If not set, all structures are imploded in a non-deterministic order (again be careful)
        :return: sitk.Image
        """
        merged_arr = None
        ref_mask = None

        if not labels:
            labels = self.list_structure_names()

        for i, label in enumerate(labels):
            i += 1
            # Instantiate merge_array
            if merged_arr is None:
                arr = sitk.GetArrayFromImage(self.get_structure(label).raster)
                merged_arr = np.zeros_like(arr)

            if ref_mask is None:
                ref_mask = self.get_structure(label)

            # Merge
            arr = sitk.GetArrayFromImage(self.get_structure(label).raster)
            merged_arr[arr.astype(bool)] = i

        img = sitk.GetImageFromArray(merged_arr)
        img.CopyInformation(ref_mask)

        return img

if __name__ == "__main__":
    ss = StructureSet(
        rtstruct_file="/home/mathis/Documents/Studies/3_GTV/documentation/end_to_end/eb552726dace/pipeline/HNCDL_003/out/segmentation.dcm",
        image_path="/home/mathis/Documents/Studies/3_GTV/documentation/end_to_end/eb552726dace/pipeline/HNCDL_003/in/1.2.840.10008.5.1.4.1.1.2/1.2.826.0.1.3680043.2.135.737080.33450775.7.1548105591.45.37/",
        raster_crop=True,
        raster_scaling_factor=10)

    print(ss.get_structure("AI_GTVt").vertices)

