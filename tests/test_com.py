import unittest
from PIL import Image
import numpy as np
from texture.analysis import CoOccur


class MyTestCase(unittest.TestCase):
    def test_offset_slices(self):
        slices = CoOccur._offset_slices(4, 225)
        self.assertEqual(slices, ([[None, -3], [3, None]], [[3, None], [None, -3]]))

        pixels = np.array([[1, 2, 3, 1, 0],
                           [0, 7, 5, 8, 2],
                           [5, 4, 0, 2, 5],
                           [7, 1, 3, 4, 9]])
        angle = 90 + 45  # ↖
        slice_start, slice_end = CoOccur._offset_slices(1, angle)
        start = pixels[slice_start[0][0]:slice_start[0][1], slice_start[1][0]:slice_start[1][1]]
        end = pixels[slice_end[0][0]:slice_end[0][1], slice_end[1][0]:slice_end[1][1]]
        self.assertEqual(start.tolist(), [[7, 5, 8, 2],
                                          [4, 0, 2, 5],
                                          [1, 3, 4, 9]])
        self.assertEqual(end.tolist(), [[1, 2, 3, 1],
                                        [0, 7, 5, 8],
                                        [5, 4, 0, 2]])

    def test_co_occur(self):
        image = Image.open("textures/1.1.04.tiff")
        co_occur = CoOccur(image, distances=[1, 2, 4, 8, 16], angles=[0, 90, 180, 270], levels=8)
        self.assertEqual(co_occur.matrices.shape, (5, 4, 8, 8))
        co_occur = CoOccur(image, distances=[1, 16], angles=[0, 120, 175.3, 240], levels=8)
        self.assertEqual(co_occur.matrices.shape, (2, 4, 8, 8))

    def test_inertia(self):
        image = Image.open("textures/1.1.04.tiff")
        l_b = np.arange(8)
        l_a = l_b[:, np.newaxis]
        coefficients = ((l_a - l_b) ** 2).reshape(1, 1, 8, 8)
        co_occur = CoOccur(image, distances=[1, 4, 8, 16, 32], angles=[0, 120, 240])
        self.assertAlmostEqual(np.sum(co_occur.matrices[2, 1] * coefficients).item(), co_occur.inertia[2, 1])
        self.assertAlmostEqual(np.sum(co_occur.matrices[4, 2] * coefficients).item(), co_occur.inertia_of(32, 240))

    def test_average(self):
        image = Image.open("textures/1.1.05.tiff")
        co_occur = CoOccur(image, distances=[1, 4, 8, 16, 32], angles=[0, 90, 240])
        self.assertAlmostEqual(np.mean(co_occur.matrices[2, :, 1, 3]).item(), co_occur.average[2, 1, 3].item())
        self.assertAlmostEqual(np.mean(co_occur.matrices[4, :, 2, 6]).item(), co_occur.average_of(32)[2, 6].item())

    def test_spread(self):
        image = Image.open("textures/1.1.10.tiff")
        co_occur = CoOccur(image, distances=[2, 5, 14], angles=[0, 32, 128, 290])
        spread_1 = np.max(co_occur.matrices[0, :, 7, 4]) - np.min(co_occur.matrices[0, :, 7, 4])
        self.assertAlmostEqual(spread_1, co_occur.spread[0, 7, 4].item())
        spread_2 = np.max(co_occur.matrices[2, :, 0, 5]) - np.min(co_occur.matrices[2, :, 0, 5])
        self.assertAlmostEqual(spread_2, co_occur.spread_of(14)[0, 5].item())


if __name__ == '__main__':
    unittest.main()
