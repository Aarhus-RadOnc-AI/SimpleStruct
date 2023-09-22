import unittest

from simplestruct.filters.hausdorff_maps import generate_hausdorff_map

from tests.utils import load_ref_and_pred


class TestHausdorffMap(unittest.TestCase):
    def setUp(self) -> None:
        self.ref_img, self.other_img = load_ref_and_pred()

    def test_hausdorff_map(self):
        hd = generate_hausdorff_map(self.ref_img, [self.other_img, self.other_img])


if __name__ == '__main__':
    unittest.main()
