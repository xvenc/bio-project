import numpy as np

class BVeinExtractor():
    def __init__(self):
        """ Override this method to initialize the preprocessor with any necessary parameters used in the preprocessing function """
        pass

    def extract(self, img_and_mask: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """ Override this method to extract veins from the given image and mask.

        Note that every class that inherits from BVeinExtractor must implement this method and must
        take image and mask as an argument and return single image representing the extracted veins.

        Args:
            img_and_mask (tuple[np.ndarray, np.ndarray]): A tuple containing the preprocessed image and mask.

        Returns:
            np.ndarray: A NumPy array representing the extracted veins.
        """
        raise NotImplementedError

    def __call__(self, image_and_mask):
        return self.extract(image_and_mask)