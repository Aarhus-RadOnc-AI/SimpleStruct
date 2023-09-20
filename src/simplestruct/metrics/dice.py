import SimpleITK as sitk

def calculate_dice(reference_image: sitk.Image, other_image: sitk.Image):
    dice = sitk.LabelOverlapMeasuresImageFilter()
    dice.Execute(reference_image, other_image)
    return dice.GetDiceCoefficient()
