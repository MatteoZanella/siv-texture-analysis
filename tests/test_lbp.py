import unittest
from texture.analysis import LBP
import numpy as np
from PIL import Image


class MyTestCase(unittest.TestCase):
    def test_neighbor_offset(self):
        offsets = LBP._neighbors_offsets(8, 1)
        expected_offsets = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]])
        self.assertEqual(offsets.tolist(), expected_offsets.tolist())

    def test_lbp(self):
        image = Image.open("textures/1.1.04.tiff")
        lbp = LBP(image)
        # lbp_2 = LBP(image, fast=False)  # Too slow
        # self.assertEqual(lbp.matrix.tolist(), lbp_2.matrix.tolist())
        self.assertEqual(lbp.matrix[0, 0], 0)
        self.assertEqual(lbp.matrix[2, 3], 199)
        self.assertEqual(lbp.matrix[0, 2], 4)

    def test_histograms(self):
        pixels = Image.fromarray(
            np.array([[1, 2, 3, 1, 0],
                     [0, 7, 5, 8, 2],
                     [5, 4, 0, 2, 5],
                     [7, 1, 3, 4, 9]]))
        lbp = LBP(pixels, cell_shape=(2, 2), neighbors=4)
        self.assertEqual(lbp.histograms[0, 0].tolist(), [.25, 0, 0, 0, .25, 0, .25, 0, 0, 0, 0, 0, 0, 0, 0, .25])
        self.assertEqual(lbp.histograms[0, 2].tolist(), [0, 0, 0, .5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .5])


if __name__ == '__main__':
    unittest.main()
