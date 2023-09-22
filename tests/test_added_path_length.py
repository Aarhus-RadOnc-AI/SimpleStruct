import unittest

import numpy as np

from simplestruct.metrics.added_path_length import APL


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

        apl_fil = APL(ref, pred)
        apl_fil.execute()
        self.assertEqual(apl_fil.get_apl(), 0.3)
        self.assertEqual(apl_fil.get_apl(normalized=False), 6)


if __name__ == '__main__':
    unittest.main()
