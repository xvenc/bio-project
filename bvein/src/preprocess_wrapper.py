import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from .preprocess.preprocessor import BVeinPreprocessor
from typing import Tuple, List

class PreprocessWrapper():
    """ A class used to load and preprocess finger vein image. """
    def __init__(self, preprocessing_functions : List[BVeinPreprocessor], separate = False) -> None:
        """ Initialize the PreprocessWrapper with a list of preprocessing functions.

        Preprocessing functions are in fact classes that implement the `BVeinPreprocessor` interface.
        They can be found in the `preprocess` directory.

        Args:
            preprocessing_functions: List of preprocessing functions to be applied in order.
            separate (bool): If True, don't chain the preprocessing steps but apply them separately (each one from original image).
        """
        self.process_func = preprocessing_functions
        self.separate = separate

    def _load_image(self, image_path: str) -> None:
        """ Load and return grayscale image given its file path. """
        return np.asarray(Image.open(image_path).convert('L'))

    def _name_from_path(self, image_path: str) -> str:
        """ Extract the image name from the file path. """
        return os.path.basename(image_path)

    def _initialize_preprocessing(self, image_path: str) -> None:
        """ Load the image and mask, and store the original image and mask in the intermediate results. """
        self.image = self._load_image(image_path)
        self.image_name = self._name_from_path(image_path)
        self.mask = np.ones_like(self.image) # Mask is the whole image by default but can be rewritten by one of the preprocessing functions
        self.process_intermediate = [(self.image, self.mask)] # Store intermediate results for plotting

    def apply_preprocessing(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """ Apply a series of image transformations to loaded image.

        Intermediate results of preprocessing steps are stored for visualization.

        Args:
            image_path (str): Path to the image file.

        Returns:
            (img, mask): A tuple containing the processed image and mask.
        """
        self._initialize_preprocessing(image_path)
        image, mask = self.image, self.mask
        for process_function in self.process_func:
            image, mask = process_function(image, mask)
            self.process_intermediate.append((image, mask))
            if self.separate:
                # Reset the image and mask to original for the next preprocessing step
                image, mask = self.image, self.mask
        return self.image, self.mask

    def _show_single_row(self, ax, image, mask, caption) -> None:
        """ Display the `image` and `mask` with `caption` in a single row defined by `ax`. """
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title(f"{caption}")
        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title(f"{caption} - mask")

    def show(self) -> None:
        """ Display the original image, mask, and intermediate preprocessing steps in a grid. """
        # How many rows to display (each row == one preprocessing step)
        rows = len(self.process_func) + 1 # +1 for the original image
        cols = 2 # Image and mask

        fig, axes = plt.subplots(rows, cols, figsize=(6, 8))
        fig.suptitle(f"{self.image_name}")

        original_data = self.process_intermediate[0]
        modified_data = self.process_intermediate[1:]

        if len(self.process_func) == 0:
            self._show_single_row(axes, *original_data, "Original")
        else:
            self._show_single_row(axes[0], *original_data, "Original")
            for i, (img, mask) in enumerate(modified_data):
                caption = self.process_func[i].__class__.__name__
                self._show_single_row(axes[i+1], img, mask, caption)

        list(map(lambda ax: ax.axis('off'), axes.flatten()))
        plt.show()

# Example usage:
if __name__ == "__main__":
    from .db import FingerVeinDatabase
    from .preprocess.lee_mask import LeeMask, ModifiedLeeMask
    from .preprocess.histogram import HistogramEqualization
    database = FingerVeinDatabase()

    # Example 1 - chain preprocessing step (each one is applied to the result of the previous one)
    preprocessing1 = [ModifiedLeeMask(mode='mode1'), HistogramEqualization()]
    iprep = PreprocessWrapper(preprocessing1)

    for img in database.get_random_batch(3)["non_target"]:
        iprep.apply_preprocessing(img)
        iprep.show()

    # Example 2 - separate preprocessing steps (each one is applied to the original image)
    preprocessing2 = [LeeMask(), ModifiedLeeMask(mode='mode1'), ModifiedLeeMask(mode='mode2')]
    iprep = PreprocessWrapper(preprocessing2, separate=True)

    for img in database.get_random_batch(3)["non_target"]:
        iprep.apply_preprocessing(img)
        iprep.show()
