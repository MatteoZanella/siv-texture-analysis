from collections import Sequence
import numpy as np
from PIL import Image


class CoOccur:
    """
    Class used to compute all Co-occurrence matrices of an image for a set of distances and angles and some related
    parameters: the inertia, the average, the spread.

    An instance of the CoOccur class holds a tensor of shape (len(distances), len(angles), levels, levels) that holds
    all Co-occurrence matrices of the image passed as constructor's parameter, for each distance and angle in the
    sequences passed as constructor's parameters. During the instantiation are computed also the inertia matrix of shape
    (len(distances), len(angles)), the average tensor of shape (len(distances), levels, levels), and the spread tensor
    of shape (len(distances), levels, levels).

    Co-occurrence matrices can dramatically grow in size, so the levels of pixels are usually quantized before the
    computation of the Co-occurrence matrices. From the 256 possible values that pixels can assume, these are reduced
    to a smaller number specified as constructor's parameter.

    Args:
        image (PIL.Image): The image that has to be analyzed, it is internally converted in B/W with Image.convert('L')
        distances (Sequence[float]): The sequence of analyzed distances. Default: range(1, 12, 2)
        angles (Sequence[float]): The sequence of analyzed angles expressed in degrees. Default: range(0, 360, 45)
        levels (int): Pixel values are quantized in this number of levels. Should be lower than 256. Default: 8
    Attributes:
        matrices (numpy.ndarray): Co-Occurrence matrices, of shape (len(distances), len(angles), levels, levels)
        inertia (numpy.ndarray): Inertia matrix, of shape (len(distances), len(angles))
        average (numpy.ndarray): Average tensor, of shape (len(distances), levels, levels)
        spread (numpy.ndarray): Spread tensor, of shape (len(distances), levels, levels)
        distances (numpy.ndarray): Array of all analyzed distances
        angles (numpy.ndarray): Array of all analyzed angles
    """
    def __init__(self, image: Image, distances: Sequence[float] = range(1, 12, 2),
                 angles: Sequence[float] = range(0, 360, 45), levels: int = 8):
        # ===Image quantization===
        pixels = np.array(image.convert('L'))  # pixels.dtype == np.uint8
        pixels = np.floor(pixels / 256 * levels).astype(np.uint8)  # quantized in the [0, levels) integer range
        # ===Angles and distances===
        self.distances = np.array(distances)
        self.angles = np.array(angles)
        self._idx_of_dist = {distance: idx for idx, distance in enumerate(distances)}
        self._idx_of_angle = {angle: idx for idx, angle in enumerate(angles)}
        # ===CoOccur Tensor=== (distances, angles, levels_start, levels_end)
        self.matrices = self._co_occurrence_matrices(pixels, distances, angles, levels)
        # ===Inertia, Average, Spread===
        self.inertia = self._inertia_matrix(levels)
        self.average = self._average_matrices()
        self.spread = self._spread_matrices()

    def _co_occurrence_matrices(self, pixels: np.ndarray, dists: Sequence, angles: Sequence, levels: int) -> np.ndarray:
        """Computes the Co-Occurrence matrix of pixels for every distance and every angle passed as parameters"""
        dists_list = []
        for distance in dists:
            angles_list = []
            for angle in angles:
                slice_start, slice_end = self._offset_slices(distance, angle)
                start = pixels[slice_start[0][0]:slice_start[0][1], slice_start[1][0]:slice_start[1][1]].reshape(-1)
                end = pixels[slice_end[0][0]:slice_end[0][1], slice_end[1][0]:slice_end[1][1]].reshape(-1)
                histogram2d = np.histogram2d(start, end, density=True, bins=levels, range=[[0, levels], [0, levels]])[0]
                angles_list.append(histogram2d)
            dists_list.append(angles_list)
        co_occur_matrices = np.array(dists_list)
        return co_occur_matrices

    @staticmethod
    def _offset_slices(distance: float, angle: float):
        """Returns the starting and ending ranges to slice the pixel matrix given an angle in degrees and a distance"""
        angle = np.radians(angle)
        offset = np.rint(np.array([-np.sin(angle), np.cos(angle)]) * distance).astype(int)
        slice_start = [[-offset[0] if offset[0] < 0 else None, -offset[0] if offset[0] > 0 else None],
                       [-offset[1] if offset[1] < 0 else None, -offset[1] if offset[1] > 0 else None]]
        slice_end = [[offset[0] if offset[0] > 0 else None, offset[0] if offset[0] < 0 else None],
                     [offset[1] if offset[1] > 0 else None, offset[1] if offset[1] < 0 else None]]
        return slice_start, slice_end

    def _inertia_matrix(self, levels: int) -> np.ndarray:
        """Returns the inertia of each distance and angle"""
        l_b = np.arange(levels)
        l_a = l_b[:, np.newaxis]
        coefficients = ((l_a - l_b) ** 2).reshape(1, 1, levels, levels)
        return np.sum(coefficients * self.matrices, axis=(-1, -2))

    def _average_matrices(self) -> np.ndarray:
        """Returns the average on all angles, for all distances and levels"""
        return np.mean(self.matrices, axis=1)

    def _spread_matrices(self) -> np.ndarray:
        """Returns the spread on all angles, for all distances and levels"""
        return np.max(self.matrices, axis=1) - np.min(self.matrices, axis=1)

    def inertia_of(self, distance: float, angle: float) -> float:
        """
        Method to get a value from the inertia matrix, relying directly on distance and angle instead of their indexes

        Args:
            distance (float): The distance of the direction
            angle (float): The angle of the direction, expressed in degrees

        Returns:
           (float): Value of the inertia matrix
        """
        return self.inertia[self._idx_of_dist[distance], self._idx_of_angle[angle]]

    def average_of(self, distance: float) -> np.ndarray:
        """
        Method to get a matrix from the average tensor, relying directly on distance instead of its index

        Args:
            distance (float): The distance of the direction

        Returns:
           (numpy.ndarray): Matrix of the average tensor
        """
        return self.average[self._idx_of_dist[distance]]

    def spread_of(self, distance: float) -> np.ndarray:
        """
        Method to get a matrix from the spread tensor, relying directly on distance instead of its index

        Args:
            distance (float): The distance of the direction

        Returns:
           (numpy.ndarray): Matrix of the spread tensor
        """
        return self.spread[self._idx_of_dist[distance]]
