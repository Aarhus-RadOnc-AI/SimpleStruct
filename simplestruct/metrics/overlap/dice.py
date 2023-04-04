import SimpleITK as sitk


def calculate_dice(referene_image: sitk.Image, other_image: sitk.Image):
    dice = sitk.LabelOverlapMeasuresImageFilter()
    dice.Execute(referene_image, other_image)
    return dice.GetDiceCoefficient()
