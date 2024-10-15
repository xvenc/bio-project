import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Import typings
from collections.abc import Callable
from typing import Tuple, NewType, List

class BVeinImage():
    """ A class used to load and preprocess finger vein image. """

    # Type hint for preprocessing functions - each function must take image and mask and return preprocessed image and mask
    PreprocessCallable = NewType('PreprocessCallable', Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]])

    def __init__(self, preprocessing_functions : List[PreprocessCallable]) -> None:
        """ Initialize the BVeinImage with a list of preprocessing functions.

        Args:
            preprocessing_functions (PreprocessCallable): List of preprocessing functions to be applied in order.
        """
        self.preprocessing_functions = preprocessing_functions
        self.processed_images = []

    def load_image(self, image_path: str) -> None:
        """ Load and return grayscale image given its file path. """
        self.image = np.asarray(Image.open(image_path).convert('L'))

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Apply a series of image transformations to loaded image.

        Returns:
            (img, mask): A tuple containing the processed image and mask.
        """

        # Mask is the whole image by default but can be rewritten by one of the preprocessing functions
        self.mask = np.ones_like(self.image) * 255

        for preprocess_fn in self.preprocessing_functions:
            self.image, self.mask = preprocess_fn(self.image, self.mask)
        return self.image, self.mask

    def show(self) -> None:
        # TODO: show original image and mask and for each preprocessing function show the processed image and mask

        # fig, axes = plt.subplots(2, 3, figsize=(10, 5))
        # fig.suptitle(f"{self.img_name}")
        # list(map(lambda ax: ax.axis('off'), axes.flatten()))
        # plt.show()
        pass
