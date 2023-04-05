import numpy as np
import SimpleITK as sitk
def scale_information_tuple(information_tuple: tuple, xy_scaling_factor: int, out_type: type, up: bool = True):
    scale_array = np.array([xy_scaling_factor, xy_scaling_factor, 1])
    if up:
        information_tuple = np.array(information_tuple) * scale_array
    else:
        information_tuple = np.array(information_tuple) / scale_array

    return tuple([out_type(info) for info in information_tuple])

def parse_dicom_image(path):
    dicom_reader = sitk.ImageSeriesReader()
    dicom_file_names = dicom_reader.GetGDCMSeriesFileNames(str(path))
    dicom_reader.SetFileNames(dicom_file_names)
    dicom_image = dicom_reader.Execute()
    return dicom_image