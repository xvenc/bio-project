import numpy as np
import cv2
from .preprocessor import BVeinPreprocessor
from typing import Tuple

class HistogramEqualization(BVeinPreprocessor):
    def __init__(self) -> None:
        """ Initialize the HistogramEqualization preprocessor. """
        pass

    def preprocess(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Apply histogram equalization to the image.

        Args:
            img (np.ndarray): The image to be equalized.
            mask (np.ndarray): The mask to be equalized.

        Returns:
            tuple: A tuple containing the equalized image and mask.
        """
        # Apply histogram equalization to the image
        equalized_img = cv2.equalizeHist(img)

        # Return the equalized image and the original mask
        return equalized_img, mask