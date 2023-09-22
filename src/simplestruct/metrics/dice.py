import SimpleITK as sitk

def dc(reference_image: sitk.Image, other_image: sitk.Image):
    dice = sitk.LabelOverlapMeasuresImageFilter()
    dice.Execute(reference_image, other_image)
    return dice.GetDiceCoefficient()
