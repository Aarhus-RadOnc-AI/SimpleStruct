import unittest

from simplestruct.filters.hausdorff_maps import HausdorffMap

from tests.utils import load_ref_and_pred


class TestHausdorffMap(unittest.TestCase):
    def setUp(self) -> None:
        self.ref_img, self.other_img = load_ref_and_pred()

    def test_hausdorff_map(self):
        HDMap = HausdorffMap(self.ref_img, [self.other_img, self.other_img, self.other_img])
        HDMap.execute()
        self.assertEqual(6, HDMap.get_hausdorff_map().shape[1])
        self.assertEqual(4, HDMap.get_summarized_hausdorff_map().shape[1])


if __name__ == '__main__':
    unittest.main()
