import numpy as np
from typing import Tuple

class BVeinExtractor():
    def __init__(self):
        """ Override this method to initialize the preprocessor with any necessary parameters used in the preprocessing function """
        pass

    def extract(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """ Override this method to extract veins from the given image and mask.

        Note that every class that inherits from BVeinExtractor must implement this method and must
        take image and mask as an argument and return single image representing the extracted veins.

        Args:
            image (np.ndarray): A preprocessed image.
            mask (np.ndarray): A preprocessed mask.

        Returns:
            np.ndarray: A NumPy array representing the extracted veins.
        """
        raise NotImplementedError

    def __call__(self, image, mask):
        return self.extract(image, mask)