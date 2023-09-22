import unittest

import numpy as np

from simplestruct.metrics.added_path_length import apl


class TestAddedPathLength(unittest.TestCase):
    def test_added_path_length(self):
        ref = np.array([
            [[0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0]]
        ])
        pred = np.array([
            [[0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0]]
        ])

        norm_apl = apl(ref, pred, True)
        self.assertEqual(norm_apl, 0.3)

        not_norm_apl = apl(ref, pred, False)
        self.assertEqual(not_norm_apl, 6)


if __name__ == '__main__':
    unittest.main()
