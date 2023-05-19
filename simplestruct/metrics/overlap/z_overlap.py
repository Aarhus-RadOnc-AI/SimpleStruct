import SimpleITK as sitk

def check_z_overlap(other_image: sitk.Image, int1: int, int2: int):
    other_arr = sitk.GetArrayFromImage(other_image)
    for z in range(other_arr.shape[0]):
        if int1 in other_arr[z, :, :] and int2 in other_arr[z, :, :]:
            return True
    else:
        return False

