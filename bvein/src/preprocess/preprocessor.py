import numpy as np
from typing import Tuple

class BVeinPreprocessor:
    def __init__(self):
        """ Override this method to initialize the preprocessor with any necessary parameters used in the preprocessing function """
        pass

    def preprocess(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Override this method to apply the preprocessing function to the image and mask

        Note that every class that inherits from BVeinPreprocessor must implement this method and must
        take image and mask as an argument and return the processed image and mask.

        Args:
            image (np.ndarray): A 2D NumPy array representing the grayscale image.
            mask (np.ndarray): A 2D NumPy array representing the mask of the image.

        Returns:
            (image, mask): A tuple containing the processed image and mask.
        """
        raise NotImplementedError

    def __call__(self, image, mask):
        return self.preprocess(image, mask)