import numpy as np
from skimage.exposure import equalize_hist, rescale_intensity
from .template import BVeinPreprocessor
from typing import Tuple

class HistogramEqualization(BVeinPreprocessor):
    def __init__(self) -> None:
        """ Initialize the HistogramEqualization preprocessor. """
        pass

    def preprocess(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Apply histogram equalization to the image.

        This code was taken from https://github.com/bioidiap/bob.bio.vein/blob/master/src/bob/bio/vein/preprocessor/filters.py

        Args:
            img (np.ndarray): The image to be equalized.
            mask (np.ndarray): The mask to be equalized.

        Returns:
            tuple: A tuple containing the equalized image and mask.
        """
        # Apply histogram equalization to the image
        retval = rescale_intensity(equalize_hist(img, mask=mask), out_range=(0, 255))

        # Make the parts outside the mask totally black
        retval[~mask] = 0

        # Return the equalized image and the original mask
        return retval, mask