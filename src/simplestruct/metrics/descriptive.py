from typing import Dict

import SimpleITK as sitk

def descriptive_stats(image: sitk.Image) -> Dict:
    f = sitk.LabelShapeStatisticsImageFilter()
    img = image != 0
    f.Execute(img)
    cc = sitk.ConnectedComponentImageFilter()
    cc.Execute(img)

    return {
        "BoundingBox": f.GetBoundingBox(1),
        "Centroid": f.GetCentroid(1),
        "Elongation": f.GetElongation(1),
        "EquivalentEllipsoidDiameter": f.GetEquivalentEllipsoidDiameter(1),
        "EquivalentSphericalPerimeter": f.GetEquivalentSphericalPerimeter(1),
        "EquivalentSphericalRadius": f.GetEquivalentSphericalRadius(1),
        "FeretDiameter": f.GetFeretDiameter(1),
        "Flatness": f.GetFlatness(1),
        "NumberOfPixels": f.GetNumberOfPixels(1),
        "PhysicalSize": f.GetPhysicalSize(1),
        "Roundness": f.GetRoundness(1),
        "ObjectCount": cc.GetObjectCount()
    }