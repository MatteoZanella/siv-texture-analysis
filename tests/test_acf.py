import unittest
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from texture.analysis import ACF


class MyTestCase(unittest.TestCase):
    def test_np_correlate_1D(self):
        vtr = np.array([2, 3, -1])
        corr = np.correlate(vtr, vtr, "full")
        self.assertListEqual(list(corr), [-2, 3, 14, 3, -2])

    def test_FFT_auto_correlate_1D(self):
        vtr = np.array([2, 3, -1])
        padded_size = 2 * vtr.size - 1
        # FFT with a padding of zeros
        vtr_ft = np.fft.fft(vtr, padded_size)
        # Pairwise multiplication and inverse transform
        corr = np.fft.ifft(np.multiply(vtr_ft, vtr_ft.conj()), padded_size)
        # Removing redundancy
        corr = corr[:vtr.size]
        # Removing the (zeroed) imaginary part and cast again to integers
        corr = np.rint(corr).real.astype(int)
        self.assertEqual(corr.tolist(), [14, 3, -2])

    def test_rFFT_auto_correlate_1D(self):
        vtr = np.array([2, 3, -1])
        padded_size = 2 * vtr.size - 1
        # Real FFT with a padding of zeros
        vtr_ft = np.fft.rfft(vtr, padded_size)
        # Pairwise multiplication and inverse transform
        corr = np.fft.irfft(np.multiply(vtr_ft, vtr_ft.conj()), padded_size)
        # Removing redundancy
        corr = corr[:vtr.size]
        # Imaginary part already removed, casting again to integers
        corr = np.rint(corr).astype(int)
        self.assertEqual(corr.tolist(), [14, 3, -2])

    def test_rFFT_auto_correlate_2D(self):
        #  2  3 -1       70 -17  0
        # -3  4  1  to  -17  21 -1
        #  5 -2  1        3  -1  2
        mtx = np.array([[2, 3, -1], [-3, 4, 1], [5, -2, 1]])
        padded_size = [2 * d - 1 for d in mtx.shape]
        # Real FFT with a padding of zeros
        mtx_ft = np.fft.rfft2(mtx, padded_size)
        # Pairwise multiplication and inverse transform
        corr = np.fft.irfft2(np.multiply(mtx_ft, mtx_ft.conj()), padded_size)
        # Removing redundancy, keeping only positive part
        corr = corr[:mtx.shape[0], :mtx.shape[1]]
        # Imaginary part already removed, casting again to integers
        corr = np.rint(corr).astype(int)
        self.assertEqual(corr.tolist(), [[70, -17, 0], [-17, 21, -1], [3, -1, 2]])

    def test_acf_constructor(self):
        # Removing negative values because pixels have values in [0,255]
        # 1 5 2      69 37  5
        # 3 4 1  to  36 27 10
        # 0 2 3      16 17  3
        image = Image.fromarray(np.array([[1, 5, 2], [3, 4, 1], [0, 2, 3]]))
        acf = ACF(image)
        expected_result = np.array([[69., 37., 5.], [36., 27., 10.], [16., 17., 3.]]) / 69
        self.assertEqual(acf.matrix.tolist(), expected_result.tolist())

    def test_eq_naive_and_fft(self):
        image = Image.open("textures/1.1.04.tiff").crop((0, 0, 128, 128))
        fft_start = time.time()
        acf_fft = ACF(image, fast=True)
        fft_end = time.time()
        acf_naive = ACF(image, fast=False)
        naive_end = time.time()
        print(f"FFT method: {fft_end - fft_start:.5}s \nNaive method: {naive_end - fft_end:.5}s")
        self.assertEqual(acf_naive.matrix.tolist(), acf_fft.matrix.tolist())

    def test_direct_mem(self):
        im_4x4 = Image.open("textures/grain_4x4.tiff")
        im_8x8 = Image.open("textures/grain_8x8.tiff")
        im_16x16 = Image.open("textures/grain_16x16.tiff")
        acf_4x4 = ACF(im_4x4)
        acf_8x8 = ACF(im_8x8)
        acf_16x16 = ACF(im_16x16)
        plt.plot(acf_4x4.matrix[0])
        plt.plot(acf_8x8.matrix[0])
        plt.plot(acf_16x16.matrix[0])
        plt.legend(['4x4', '8x8', '16x16'])
        plt.show()
        self.assertEqual(acf_4x4.directional_memory(), (32, 32))
        self.assertEqual(acf_8x8.directional_memory(), (16, 16))
        self.assertEqual(acf_16x16.directional_memory(), (8, 8))

    def test_mean_val(self):
        # 1 5 2      69 37  5
        # 3 4 1  to  36 27 10
        # 0 2 3      16 17  3
        image = Image.fromarray(np.array([[1, 5, 2], [3, 4, 1], [0, 2, 3]]))
        acf = ACF(image)
        expected_result = (145 / 69, 117 / 69)
        self.assertAlmostEqual(acf.mean_values[0], expected_result[0])
        self.assertAlmostEqual(acf.mean_values[1], expected_result[1])

    def test_moments(self):
        # 1 5 2      69 37  5
        # 3 4 1  to  36 27 10
        # 0 2 3      16 17  3
        image = Image.fromarray(np.array([[1, 5, 2], [3, 4, 1], [0, 2, 3]]))
        acf = ACF(image)
        self.assertAlmostEqual(acf.moment(1, 1), 5.58250458)
        self.assertAlmostEqual(acf.moment(2, 2), 15.94700232)


if __name__ == '__main__':
    unittest.main()
