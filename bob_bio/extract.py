import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from bob.bio.vein.extractor import RepeatedLineTracking
from bob.bio.vein.preprocessor import LeeMask, FixedCrop, HuangNormalization, HistogramEqualization

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

    def extract(self, extractor, img_and_mask):
        """Extract veins from the given image and mask using the provided extractor function.

        Extractor function must be one of the classes defined in bio.bob.vein, which can be found at:
        https://www.idiap.ch/software/bob/docs/bob/docs/stable/bob/bob.bio.vein/doc/api.html#module-bob.bio.vein.extractor

        Args:
            extractor (callable): A callable class that takes an image and mask and returns the extracted veins.
            img_and_mask (tuple): A tuple containing the preprocessed image and mask.

        Returns:
            veins: Extracted veins as a 2D NumPy array.
        """
        print("Extracting veins...")
        self.img_and_mask = img_and_mask
        self.veins = extractor(img_and_mask)
        return self.veins

    def show_veins(self, img_name):
        """Display the preprocessed image, mask, and extracted veins in a 1x3 grid using Matplotlib.

        Args:
            img_name (str): The name of the image to be displayed as the overall title.
        """
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        fig.suptitle(f"{img_name}")

        axes[0].imshow(self.img_and_mask[0], cmap='gray')
        axes[0].set_title("Preprocessed Image")

        axes[1].imshow(self.img_and_mask[1], cmap='gray')
        axes[1].set_title("Preprocessed Mask")

        axes[2].imshow(self.veins, cmap='gray')
        axes[2].set_title("Extracted Veins")

        list(map(lambda ax: ax.axis('off'), axes.flatten()))
        plt.show()

def preprocess_1():
    """
    This preprocessing function uses fixed cropping to remove the top and bottom 30 pixels of the image,
    Lee mask to extract the mask of the finger, Huang normalization to normalize the image and mask (center them),
    and histogram equalization to enhance the contrasts of the image.
    """
    cropper = FixedCrop(top=30, bottom=30)
    masker = LeeMask()
    normalizer = HuangNormalization()
    filter = HistogramEqualization()
    return cropper, masker, normalizer, filter

def extract_1():
    """
    This extractor is a simple repeated line tracking algorithm that uses 1000 iterations and does not rescale the image.
    https://www.idiap.ch/software/bob/docs/bob/docs/stable/bob/bob.bio.vein/doc/api.html#module-bob.bio.vein.extractor
    """
    return RepeatedLineTracking(iterations=1000, rescale=False)

if __name__ == "__main__":
    # Preprocessing transformations (uncomment one)
    T = preprocess_1()

    # Vein extraction algorithm (uncomment one)
    E = extract_1()

    # Examples finger images to process
    IMGS = [
        "bestcase.bmp",     # Best lighting
        "basic.bmp",        # Basic image
        "uncentered.bmp",   # Image where the finger is not centered
    ]
    IMGS = [f"vein_imgs/{img}" for img in IMGS]
    procimg = ImagePreprocessor()
    extractor = VeinExtractor()

    # Show picked transformations for each image
    processed = []
    for img in IMGS:
        procimg.load_image(img)
        processed_data = procimg.preprocess(*T)
        # procimg.show_all_transformations()
        processed.append((img, processed_data)) # Store the processed data for vein extraction

    # Extract and display extracted veins for each preprocessed image
    for (img_name, img_and_mask) in processed:
        extractor.extract(E, img_and_mask)
        extractor.show_veins(img_name)
