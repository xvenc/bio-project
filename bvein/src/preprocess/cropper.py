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

class MaskCropper(BVeinPreprocessor):
    """ A preprocessor that crops the image and mask based on the mask's content. """

    def _crop_parameters_from_mask(self, mask: np.ndarray) -> int:
        """ Find ideal crop parameters based on the mask's content.

        Args:
            mask (np.ndarray): The mask to be used for cropping.

        Returns:
            start_y: starting y coordinate for cropping
        """
        max_height = 0              # Maximum height of the mask
        min_height = mask.shape[0]  # Minimum height of the mask
        top_max_y = 0               # y coordinate of the top part of image where the mask heigth is the highest
        top_min_y = 0               # y coordinate of the top part of image where the mask height is lowest
        bottom_max_y = 0            # y coordinate of the bottom part of image where the mask is highest
        bottom_min_y = 0            # y coordinate of the bottom part of image where the mask is lowest

        for column in mask.T:
            mask_nonzero = np.nonzero(column)[0]  # Get indices of where the mask in non-zero
            mask_height = np.size(mask_nonzero)   # Mask height at this column
            if mask_height > max_height:
                # We found a column with higher mask height, store it and its y coordinates
                max_height = mask_height
                top_max_y = mask_nonzero[0]
                bottom_max_y = mask_nonzero[-1]
            elif mask_height < min_height:
                # We found a column with lower mask height, store it and its y coordinates
                min_height = mask_height
                top_min_y = mask_nonzero[0]
                bottom_min_y = mask_nonzero[-1]

        # At this point we have the y coordinates of the top and bottom parts of the image where the mask is highest and lowest
        # Take middle of each pair (top_max_y, top_min_y) and (bottom_max_y, bottom_min_y)
        top_midpoint = (top_min_y - top_max_y) // 2
        bottom_midpoint = (bottom_max_y - bottom_min_y) // 2
        # Now calculate the cropping parameters
        crop_start = top_max_y + top_midpoint
        crop_end = mask.shape[0] - (bottom_max_y - bottom_midpoint)
        return crop_start, crop_end

    def preprocess(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Crop the image and mask based on the mask's content. """
        crop_start, crop_end = self._crop_parameters_from_mask(mask)
        c = Cropper(crop_start, crop_end, 0, 0)
        return c.preprocess(image, mask)