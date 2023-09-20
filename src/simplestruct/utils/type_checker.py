import numpy as np
import SimpleITK as sitk

def is_array(obj):
    return isinstance(obj, np.ndarray)
def is_image(obj):
    return isinstance(obj, sitk.Image)