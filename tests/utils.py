import os
import zipfile
from io import BytesIO
from typing import Union

import SimpleITK as sitk
import numpy as np
import requests


def get_test_dicom(path, url):
    res = requests.get(url)
    res_io = BytesIO(res.content)
    zf = zipfile.ZipFile(file=res_io, compression=zipfile.ZIP_DEFLATED)
    zf.extractall(path)
    return path


def load_ref_and_pred():
    reference_path = "tests/data/HN1004_20190403_CT/scans/1-unknown/resources/NIFTI/files/mask_GTV-1.nii.gz"
    other_path = "tests/data/HN1004_20190403_CT/scans/1-unknown/resources/NIFTI/files/mask_GTV-1.nii.gz"
    try:

        ref_img = sitk.ReadImage(reference_path)
        other_img = sitk.ReadImage(other_path)
    except:
        test_data = "./tests/data/"
        os.makedirs(test_data)
        get_test_dicom(path=test_data,
                       url="https://xnat.bmia.nl/REST/projects/stwstrategyhn1/subjects/BMIAXNAT_S09203/experiments/BMIAXNAT_E62311/scans/1/files?format=zip")

        ref_img = sitk.ReadImage(reference_path)
        other_img = sitk.ReadImage(other_path)

    return ref_img, other_img