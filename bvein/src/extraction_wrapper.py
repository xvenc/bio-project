import matplotlib.pyplot as plt
import numpy as np

# Import typings
from collections.abc import Callable
from typing import List

class ExtractionWrapper():
    """ A class used to extract and display veins from preprocessed images and masks. """
    def __init__(self, extractor_functions : Callable) -> None:
        """ Initialize the ExtractionWrapper with a list of extractor functions.

        Args:
            extractor_functions (Callable): List of callable classes implementing vein extraction methods to be applied.
        """
        self.extractor_funcs = extractor_functions
        self.extractor_names = [ef.__class__.__name__ for ef in extractor_functions]

    def extract(self, image: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
        """ Extract veins from the given image and mask using the provided extractor functions.

        Args:
            image (np.ndarray): A preprocessed image.
            mask (np.ndarray): A preprocessed mask.

        Returns:
            list: A list of extracted veins as 2D NumPy arrays.
        """
        # Store original image and mask for visualization
        self.image = image
        self.mask = mask
        self.extracted_veins_imgs = [extractor(image, mask) for extractor in self.extractor_funcs]
        return self.extracted_veins_imgs

    def show(self) -> None:
        """ Display the preprocessed image, mask, and extracted veins in a grid. """
        ext_len = len(self.extractor_names)
        _, axes = plt.subplots(1 + (ext_len // 2) + (ext_len % 2), 2, figsize=(8, 6))

        axes[0][0].imshow(self.image, cmap='gray')
        axes[0][0].set_title("Preprocessed Image")
        axes[0][1].imshow(self.mask, cmap='gray')
        axes[0][1].set_title("Preprocessed Mask")

        for i, (extracted_veins_img, ext_name) in enumerate(zip(self.extracted_veins_imgs, self.extractor_names)):
            axes[i // 2 + 1][i % 2].imshow(extracted_veins_img, cmap='gray')
            axes[i // 2 + 1][i % 2].set_title(f"{ext_name} Veins")

        list(map(lambda ax: ax.axis('off'), axes.flatten()))
        plt.show()
