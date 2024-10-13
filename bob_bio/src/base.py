import matplotlib.pyplot as plt
import numpy as np
import bob.bio.vein.extractor as be
import bob.bio.vein.preprocessor as bp
from PIL import Image

class ImagePreprocessor():
    """A class used to represent and process an image through various transformations."""
    def __init__(self):
        pass

    def load_image(self, img_path):
        """Load the image from the given path as a grayscale image."""
        self.img = np.asarray(Image.open(img_path).convert('L'))
        self.img_name = img_path.split("/")[-1]
        return self.img

    def preprocess(self, croper, masker, normalizer, filter):
        """Apply a series of preprocessing steps to the image.

        All possible preprocessing steps are described in bio.bob.vein documentation, which can be found at:
        https://www.idiap.ch/software/bob/docs/bob/docs/stable/bob/bob.bio.vein/doc/api.html
        Alternatively, you can see the implementation of these preprocessing steps at their GitHub:
        https://github.com/bioidiap/bob.bio.vein/blob/master/src/bob/bio/vein/preprocessor

        Args:
            croper (callable): Crops the image.
            masker (callable): Extracts the mask of the finger from the image.
            normalizer (callable): Normalizes the image and mask.
            filter (callable): Applies additional filtering to the image.

        Returns:
            tuple: A tuple containing the filtered image and the normalized mask.
        """
        self._cropped_img = croper(self.img)
        self._mask = masker(self._cropped_img)
        self._norm_img, self._norm_mask = normalizer(self._cropped_img, self._mask)
        self._filtered_img = filter(self._norm_img, self._norm_mask)
        return self._filtered_img, self._norm_mask

    def show_all_transformations(self):
        """Display a series of image transformations in a 2x3 grid using Matplotlib."""
        fig, axes = plt.subplots(2, 3, figsize=(10, 5))
        fig.suptitle(f"{self.img_name}")

        axes[0][0].imshow(self.img, cmap='gray')
        axes[0][0].set_title("Original Image")

        axes[0][1].imshow(self._cropped_img, cmap='gray')
        axes[0][1].set_title("Cropped Image")

        axes[0][2].imshow(self._mask, cmap='gray')
        axes[0][2].set_title("Mask")

        axes[1][0].imshow(self._norm_img, cmap='gray')
        axes[1][0].set_title("Normalized Image")

        axes[1][1].imshow(self._norm_mask, cmap='gray')
        axes[1][1].set_title("Normalized Mask")

        axes[1][2].imshow(self._filtered_img, cmap='gray')
        axes[1][2].set_title("Filtered Image")

        list(map(lambda ax: ax.axis('off'), axes.flatten()))
        plt.show()

class VeinExtractor():
    """A class used to extract and display veins from preprocessed images and masks."""
    def __init__(self):
        pass

    def extract(self, extractors, img_and_mask):
        """Extract veins from the given image and mask using the provided extractor functions.

        Extractor functions must be one of the classes defined in bio.bob.vein, which can be found at:
        https://www.idiap.ch/software/bob/docs/bob/docs/stable/bob/bob.bio.vein/doc/api.html#module-bob.bio.vein.extractor

        Args:
            extractors (list): A list of callable classes that take an image and mask and return the extracted veins.
            img_and_mask (tuple): A tuple containing the preprocessed image and mask.

        Returns:
            list: A list of extracted veins as 2D NumPy arrays.
        """
        self.img_and_mask = img_and_mask
        self.ext_names = [ext.__class__.__name__ for ext in extractors]

        # print("Extracting veins using the following extractors:", self.ext_names)
        self.extracted_veins_imgs = [extractor(img_and_mask) for extractor in extractors]

        return self.extracted_veins_imgs

    def show_veins(self, img_name):
        """Display the preprocessed image, mask, and extracted veins in a grid using Matplotlib.

        Args:
            img_name (str): The name of the image to be displayed as the overall title.
        """
        ext_len = len(self.ext_names)

        fig, axes = plt.subplots(1 + (ext_len // 2) + (ext_len % 2), 2, figsize=(8, 6))
        fig.suptitle(f"{img_name}")

        axes[0][0].imshow(self.img_and_mask[0], cmap='gray')
        axes[0][0].set_title("Preprocessed Image")
        axes[0][1].imshow(self.img_and_mask[1], cmap='gray')
        axes[0][1].set_title("Preprocessed Mask")

        for i, (extracted_veins_img, ext_name) in enumerate(zip(self.extracted_veins_imgs, self.ext_names)):
            axes[i // 2 + 1][i % 2].imshow(extracted_veins_img, cmap='gray')
            axes[i // 2 + 1][i % 2].set_title(f"{ext_name} Veins")

        list(map(lambda ax: ax.axis('off'), axes.flatten()))
        plt.show()


# Preprocessing functions
def preprocess_1():
    """
    This preprocessing function uses fixed cropping to remove the top and bottom 30 pixels of the image,
    Lee mask to extract the mask of the finger (I found this mask to work the best), no normalization, and no additional filtering.
    """
    cropper = bp.FixedCrop(top=30, bottom=30)
    masker = bp.LeeMask()
    normalizer = bp.NoNormalization()
    filter = bp.NoFilter()
    return cropper, masker, normalizer, filter

def preprocess_2():
    """
    Same as preprocess_1 but with histogram equalization to enhance the contrasts of the image.
    """
    cropper = bp.FixedCrop(top=30, bottom=30)
    masker = bp.LeeMask()
    normalizer = bp.NoNormalization()
    filter = bp.HistogramEqualization()
    return cropper, masker, normalizer, filter

def preprocess_3():
    """
    Same as preprocess_2 but with Huang normalization to normalize the image and mask (which should center them).
    """
    cropper = bp.FixedCrop(top=30, bottom=30)
    masker = bp.LeeMask()
    normalizer = bp.HuangNormalization()
    filter = bp.HistogramEqualization()
    return cropper, masker, normalizer, filter


# Vein extraction functions
def extract_rtl():
    """
    This extractor is a simple repeated line tracking algorithm that uses 1000 iterations and does not rescale the image.
    https://www.idiap.ch/software/bob/docs/bob/docs/stable/bob/bob.bio.vein/doc/api.html#bob.bio.vein.extractor.RepeatedLineTracking
    """
    # lower number of iterations because 3000 produce thick lines
    return be.RepeatedLineTracking(iterations=1000, rescale=False)

def extract_mc():
    """
    This extractor is a simple maximum curvature algorithm.
    https://www.idiap.ch/software/bob/docs/bob/docs/stable/bob/bob.bio.vein/doc/api.html#bob.bio.vein.extractor.MaximumCurvature
    """
    # lower the std. deviation of the Gaussian filter
    # produces more separated dots around the main vein lines
    return be.MaximumCurvature(sigma=3)

def extract_wl():
    """
    This extractor is a wide line detector algorithm.
    https://www.idiap.ch/software/bob/docs/bob/docs/stable/bob/bob.bio.vein/doc/api.html#bob.bio.vein.extractor.WideLineDetector
    """
    # Cant get this to work nicely
    return be.WideLineDetector(threshold=3,g=35,rescale=False)

def extract_pc():
    """
    This extractor is a principal curvature algorithm.
    https://www.idiap.ch/software/bob/docs/bob/docs/stable/bob/bob.bio.vein/doc/api.html#bob.bio.vein.extractor.PrincipalCurvature
    """
    return be.PrincipalCurvature(sigma=3, threshold=4)