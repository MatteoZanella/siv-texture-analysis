import numpy as np
from PIL import Image


class ACF:
    """
    Class used to compute the autocorrelation function of an image and some related statistical parameters.
    It computes the autocorrelation matrix and texture parameters of directional memory, mean values, moments.

    An instance of the ACF class holds the ACF matrix of the image used as constructor's parameter, computed during the
    instantiation. Moreover, during the instantiation also the mean values are computed and saved as an attribute.

    Args:
        image (PIL.Image): The image that has to be analyzed, it is internally converted in B/W with Image.convert('L')
        normalize (bool): If the matrix should be normalized with the element in position [0,0]. Default: True
        fast (bool): If Fast Fourier Transform method should be used, or the slowest naive implementation. Default: True

    Attributes:
        matrix (numpy.ndarray): The autocorrelation function matrix computed during the ACF class instantiation
            upon a specified image.
        mean_values (float, float): The mean values of the autocorrelation function matrix
    """

    def __init__(self, image: Image, normalize=True, fast=True):
        pixels = np.array(image.convert('L'))  # pixels.dtype == np.uint8
        # ===ACF Matrix===
        if fast:
            acf_matrix = self._fft_autocorrelation(pixels)
        else:
            acf_matrix = self._naive_autocorrelation(pixels)
        self.matrix = (acf_matrix / acf_matrix[0, 0]) if normalize else acf_matrix
        # ===Mean values===
        self.mean_values = self._mean_values()

    @staticmethod
    def _fft_autocorrelation(pixels: np.ndarray) -> np.ndarray:
        """
        ACF computed using real Fast Fourier Transform.

        Args:
            pixels (np.ndarray): The 2D matrix of pixels used to compute the ACF

        Returns:
            (np.ndarray): The ACF matrix
        """
        padded_size = [2 * d - 1 for d in pixels.shape]
        # Real FFT with a padding of zeros, because acf spans on a doubled size
        pixels_freq = np.fft.rfft2(pixels, padded_size)
        # Autocorrelation in the frequency domain
        acf_freq = np.multiply(pixels_freq, pixels_freq.conj())
        # Going back to the space domain with the inverse FFT
        acf = np.fft.irfft2(acf_freq, padded_size)
        # Removing redundancy, keeping only positive shifts
        acf = acf[:pixels.shape[0], :pixels.shape[1]]
        # Imaginary part already removed by conjugate multiplication, casting again to integers
        acf = np.rint(acf)
        # Normalization
        return acf

    @staticmethod
    def _naive_autocorrelation(pixels: np.ndarray) -> np.ndarray:
        """
        ACF computed with simple sums and multiplications, following the definition.

        Args:
            pixels (np.ndarray): The 2D matrix of pixels used to compute the ACF

        Returns:
            (np.ndarray): The ACF matrix
        """
        acf = np.empty(pixels.shape)
        pixels = pixels.astype(np.float64)  # Convert to avoid overflows
        i_dim = acf.shape[0]
        j_dim = acf.shape[1]
        for i_shift in range(0, i_dim):
            for j_shift in range(0, j_dim):
                acf[i_shift, j_shift] = np.sum(pixels[i_shift:, j_shift:] * pixels[:i_dim - i_shift, :j_dim - j_shift])
        return acf

    def _mean_values(self) -> (float, float):
        """
        Mean values computed on the instance ACF matrix.

        Returns:
            m_mean, n_mean (float, float): The mean values in the first and second axis

        Notes:
            These values can be used multiple times in the computation of moments, so their value is computed in the
            instantiation and stored in a class attribute.
        """
        # sum_m(m * sum_n(matrix))
        m = np.arange(self.matrix.shape[0])
        m_mean = np.sum(m * np.sum(self.matrix, axis=1))
        # sum_n(n * sum_m(matrix))
        n = np.arange(self.matrix.shape[1])
        n_mean = np.sum(n * np.sum(self.matrix, axis=0))
        return m_mean, n_mean

    def moment(self, p: int, q: int) -> float:
        """
        Function to compute all possible moments of order p, q of the ACF:
        SUM_m(SUM_n( (m-m_mean)^p * (n-n_mean)^q * acf(m,n) ))

        Args:
            q (int): exponent referring to the first axis m
            p (int): exponent referring to the second axis n

        Returns:
            M_pq (float): the moment of order p,q of the ACF

        Notes:
            The implementation moves the (n-n_mean)^q outside the SUM_m to reduce computations, resulting in the
            following: SUM_n( (n-n_mean)^q * SUM_m( (m-m_mean)^p * acf(m,n) ))
        """
        m_mean, n_mean = self.mean_values
        p_factors = ((np.arange(self.matrix.shape[0]) - m_mean) ** p).reshape(-1, 1)  # Reshape as a column
        q_factors = (np.arange(self.matrix.shape[1]) - n_mean) ** q
        # Broadcast multiplication of p_factors columns on whole matrix
        # Sum the columns together, obtaining a single row
        p_sum = np.sum(p_factors * self.matrix, axis=0)
        # Multiplication of the q_factors row with the p_sum row
        # Sum the row, obtaining a single value
        return np.sum(q_factors * p_sum).item()

    def profile_spreads(self) -> (float, float):
        """
        Function to computed the profile spreads, equivalent to call moment(2, 0), moment(0, 2).

        Returns:
            M_20, M_01 (float, float): the profile spreads of the ACF
        """
        return self.moment(2, 0), self.moment(0, 2)

    def cross_correlation(self) -> float:
        """
        Function to computed the cross correlation, equivalent to call moment(1, 1).

        Returns:
            M_11 (float): the cross correlation of the ACF
        """
        return self.moment(1, 1)

    def second_deg_spreads(self) -> float:
        """
        Function to computed the 2nd-degree spreads, equivalent to call moment(2, 2).

        Returns:
            M_22 (float): the 2nd-degree spreads of the ACF
        """
        return self.moment(2, 2)

    def directional_memory(self) -> (int, int):
        """
        Computes the directional memory of the autocorrelation matrix. It provides a measure of the granularity of the
        texture: a rough texture will have shorter memory with respect to a smoother one.

        The method finds the first occurrence the values is <=0.5  for [m_x, 0], [0, m_y] rows

        Returns:
            m_x, m_y (int, int): The two indices such that acf.matrix(m_x,0) ≈ acf.matrix(0,m_y) ≈ 0.5
        """
        x_line = self.matrix[:, 0]
        y_line = self.matrix[0, :]

        # First occurrence where the acf value goes below or equal .5
        m_x = np.argmax(x_line <= .5)
        m_y = np.argmax(y_line <= .5)

        return m_x, m_y
