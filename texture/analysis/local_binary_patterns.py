import warnings
import math
import numpy as np
from PIL import Image


class LBP:
    """
    Class used to compute the Local Binary Patterns of an image and the resulting histograms.
    It computes the matrix holding the local representation of the texture for each pixel, i.e. the comparison of each
    pixel with its surrounding neighbours. Then from this matrix it can compute the normalized histogram of each cell
    and the resulting feature vector obtained concatenating each cell's histogram

    An instance of the LBP class holds the LBP matrix of the image used as constructor's parameter, computed during the
    instantiation. The image is padded with zeros, for computing values on image edges.

    The number of neighbours and their distance from the considered pixel can be specified as parameters, the offset
    values are computed as equidistant points on a circumference, in clockwise direction starting from the middle-up
    position, and approximated to the nearest integers in order to obtain the coordinates on the pixels matrix.
    The number of neighbors influence the size of words (neighbors-bit words) and the size of the histogram, which range
    is [0, 2^neighbor - 1].

    For the histogram computation, the image is divided in the certain number of cells which shape is specified as
    parameter. If the cells don't perfectly tile the matrix, the ones on the right and bottom border will be shrunken
    and a warning will be displayed. The histogram of each cell is computed and normalized. A matrix of all histograms
    organized as the cell's placement is saved as an attribute, together with a feature vector representing a view on
    the flattened concatenation of all histograms.

    Args:
        image (PIL.Image): The image that has to be analyzed, it is internally converted in B/W with Image.convert('L')
        cell_shape (int, int): The dimension of each cell in the LBP matrix on which the histograms are computed.
            Default: (16, 16)
        neighbors (int): The number of neighbors considered for each pixel. Default: 8
        distance (int): The distance of the considered neighbors. Default: 1
        fast (bool): If true, uses a numpy-efficient algorithm instead of the definition-like one. Default: True

    Attributes:
        matrix (numpy.ndarray): The Local Binary Patterns matrix computed during the LBP class instantiation
            upon a specified image
        histograms (numpy.ndarray): A matrix containing the cells' histograms, organized as the cells are arranged in
            the Local Binary Patterns matrix
        feature_vtr (numpy.ndarray): A view of the histograms matrix that concatenates all histograms in a single
            vector. The histograms are ordered row by row, from left to right.
            Equivalent to LBP(...).histograms.reshape(-1).
        hist_bins (int): Number of bins, or buckets, used for each histogram. It is the maximum value that a
            pixel's word can represent for the given number of neighbors specified
        neighbors_offsets (numpy.ndarray): A matrix holding in the i-th row the row and column offsets needed to reach
            the i-th neighbor of a pixel

    """

    def __init__(self, image: Image, cell_shape=(16, 16), neighbors=8, distance=1, fast=True):
        pixels = np.array(image.convert('L'))  # pixels.dtype == np.uint8
        # ===Histogram bins===
        self.hist_bins = 2 ** neighbors
        # ===Neighbor offsets===
        self.neighbors_offsets = self._neighbors_offsets(neighbors, distance)
        # ===LBP Matrix===
        if fast:
            self.matrix = self._local_binary_patterns(pixels, distance)
        else:
            self.matrix = self._naive_local_binary_patterns(pixels, distance)
        # ===Histograms===
        self.histograms = self._histograms(cell_shape)
        self.feature_vtr = self.histograms.reshape(-1)

    def _local_binary_patterns(self, pixels: np.ndarray, distance: int) -> np.ndarray:
        """Returns the Local Binary Patters matrix of the given pixels"""
        # # The dtype of the lbp_matrix follows the maximum size of the integers it can contain. Defaults to uint8
        # lbp_matrix = np.empty(pixels.shape, self._best_dtype(self.histogram_bins))
        # The pixel matrix is padded with with zeros so neighbors of pixels placed in the borders can be referenced
        padded_pixels = np.pad(pixels, distance, constant_values=0)
        # Compute the 3D matrix holding all neighbors for each position
        neighbors = np.stack([self._shift(padded_pixels, offset, distance) for offset in self.neighbors_offsets])
        neighbors = np.moveaxis(neighbors, 0, -1)  # Rearrange the shape to be (m x n x neighbors)
        # If the neighbor is greater or equal -> 1 (True); if the the center is greater -> 0 (False)
        words = neighbors >= pixels[..., np.newaxis]
        # Transform the binary arrays to a decimal value
        lbp_matrix = self._words_to_decimals(words)
        # The dtype follows the maximum size of the integers it can contain. With the default 8 neighbors, it is uint8
        return lbp_matrix.astype(self._best_dtype(self.hist_bins))

    def _histograms(self, cell_shape: (int, int)) -> np.ndarray:
        """Returns the histograms matrix on the LBP matrix attribute divided in cells of shape specified as parameter"""
        # Raise a warning if the image can't be evenly divided and some cells have to be smaller
        celled_shape = self.matrix.shape[0] / cell_shape[0], self.matrix.shape[1] / cell_shape[1]
        if not celled_shape[0].is_integer() or not celled_shape[0].is_integer():
            warnings.warn(f"The cell's shape {cell_shape} cannot perfectly tile the image's shape {self.matrix.shape}",
                          stacklevel=3)
        celled_shape = tuple(int(math.ceil(dim)) for dim in celled_shape)
        # List that every (n, n+1) couple represents the indices range of a cell
        rows = [None, *range(cell_shape[0], self.matrix.shape[0], cell_shape[0]), None]  # [None 16 32 ... 496 None]
        cols = [None, *range(cell_shape[1], self.matrix.shape[1], cell_shape[1]), None]
        histograms = np.array([[
            np.histogram(self.matrix[rows[i]:rows[i + 1], cols[j]:cols[j + 1]],
                         bins=self.hist_bins, range=(0, self.hist_bins), density=True)[0]
            for j in range(0, celled_shape[1])]
            for i in range(0, celled_shape[0])])
        return histograms

    @staticmethod
    def _neighbors_offsets(neighbors: int, distance: int) -> np.ndarray:
        """
        Function to compute the offsets of a certain number of neighbors, at a given distance. The method finds the
        desired number of equidistant neighbors on a circumference, starting from -π/2 in a counterclockwise direction
        because in the matrix indexing going upwards means to decrease the index. The resulting offsets give the
        neighbors in a clockwise direction starting from exactly above the center pixel.

        Args:
            neighbors (int): The number of neighbors considered
            distance (int): The distance from the center pixel to the neighbors

        Returns:
            A matrix holding in the i-th row the row and column offsets needed to reach the i-th neighbor
        """
        angles = - np.pi / 2 + np.arange(0, neighbors) * 2 * np.pi / neighbors
        offsets = np.stack([np.sin(angles), np.cos(angles)], axis=1) * distance
        # Rounding coordinates
        return np.rint(offsets).astype(int)

    @staticmethod
    def _shift(padded_pixels: np.ndarray, offset: np.ndarray, padding: int) -> np.ndarray:
        """Takes the padded matrix and returns the view on the original matrix's position and shifted of the offset"""
        coord = np.array([[padding, padding], np.array(padded_pixels.shape) - padding]) + offset
        return padded_pixels[coord[0, 0]:coord[1, 0], coord[0, 1]:coord[1, 1]]

    @staticmethod
    def _words_to_decimals(words: np.ndarray) -> np.ndarray:
        """Function to compute the decimal value of an array of bits ordered from the most to the least significant"""
        coefficients = (1 << np.arange(words.shape[-1])[::-1])  # [..., 128, 64, 32, 16, 8, 4, 2, 1]
        return np.tensordot(words, coefficients, axes=([2], [0]))

    @staticmethod
    def _best_dtype(max_value: int) -> np.dtype:
        """Returns the best dtype for the LBP matrix, to save memory space """
        neighbors = np.log2(max_value)
        if neighbors <= 8:
            bits = np.uint8
        elif neighbors <= 16:
            bits = np.uint16
        elif neighbors <= 32:
            bits = np.uint32
        else:
            bits = np.uint64
        return np.dtype(bits)

    def _naive_local_binary_patterns(self, pixels: np.ndarray, distance: int) -> np.ndarray:
        # Slow implementation: ~6.2s on a 512x512 matrix
        lbp_matrix = np.empty(pixels.shape)
        # the pixel matrix is padded with with zeros so neighbors of pixels placed in the borders can be referenced
        padded_pixels = np.pad(pixels, distance, constant_values=0)
        for i in range(0, lbp_matrix.shape[0]):
            for j in range(0, lbp_matrix.shape[1]):
                neighbors_coord = (self.neighbors_offsets + [i, j] + distance).T
                # Passing the rows and then the columns of the coordinates
                neighbors = padded_pixels[neighbors_coord[0], neighbors_coord[1]]
                # If the neighbor is greater or equal -> 1 (True); if the the center is greater -> 0 (False)
                word = np.greater_equal(neighbors, pixels[i, j])
                lbp_matrix[i, j] = word.dot(1 << np.arange(word.size)[::-1])
        return lbp_matrix
