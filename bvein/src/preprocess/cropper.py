import numpy as np
from .preprocessor import BVeinPreprocessor
from typing import Tuple

class Cropper(BVeinPreprocessor):
    def __init__(self, top: int, bottom: int, left: int, right: int) -> None:
        """ Initialize the Cropper with the desired crop size.

        Args:
            top (int): Number of pixels to crop from the top.
            bottom (int): Number of pixels to crop from the bottom.
            left (int): Number of pixels to crop from the left.
            right (int): Number of pixels to crop from the right.
        """
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def preprocess(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Crop the image and mask by the specified amount of pixels.

        Args:
            img (np.ndarray): The image to be cropped.
            mask (np.ndarray): The mask to be cropped.

        Returns:
            tuple: A tuple containing the cropped image and mask.
        """
        # Crop the image using array slicing
        cropped_img = img[self.top:img.shape[0] - self.bottom, self.left:img.shape[1] - self.right]

        # Crop the mask using the same dimensions
        cropped_mask = mask[self.top:mask.shape[0] - self.bottom, self.left:mask.shape[1] - self.right]

        return cropped_img, cropped_mask