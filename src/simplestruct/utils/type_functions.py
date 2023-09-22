from typing import Union

import numpy as np
import SimpleITK as sitk

def is_array(obj):
    return isinstance(obj, np.ndarray)
def is_image(obj):
    return isinstance(obj, sitk.Image)


def load_as_np_array(image: Union[sitk.Image, np.ndarray]):
    if is_image(image):
        return sitk.GetArrayFromImage(image)
    else:
        return image