import SimpleITK as sitk
class DescriptiveStats:

    def __init__(self, image: sitk.Image):
        self.image = image != 0  # Binarize
        self.stats = None
    def execute(self):
        f = sitk.LabelShapeStatisticsImageFilter()
        f.Execute(self.image)
        cc = sitk.ConnectedComponentImageFilter()
        cc.Execute(self.image)
        self.stats = {
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
            "ObjectCount": cc.GetObjectCount(),
            "Spacing": self.image.GetSpacing(),
            "Size": self.image.GetSize(),
            "Origin": self.image.GetOrigin(),
            "Direction": self.image.GetDirection(),

        }
    def get_descriptive_stats(self):
        if self.stats is None:
            self.execute()
        return self.stats