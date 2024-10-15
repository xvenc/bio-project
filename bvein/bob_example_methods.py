import src.db as db
import src.image as image
import src.extractor as extractor
import matplotlib.pyplot as plt

import bob.bio.vein.preprocessor as bp
import bob.bio.vein.extractor as be

class BobVeinImage(image.BVeinImage):
    def __init__(self):
        super().__init__([])

    # We need to overwrite preprocess since bob applies transformations in a specific order
    # and each transformation takes different parameters and returns diffrerent values
    # e.g some require image and mask, some only image, some return only image, some return image and mask
    def preprocess(self, croper, masker, normalizer, filter):
        """ Apply a series of preprocessing steps to the image based on bob library.

        Args:
            croper (callable): Crops the image - does not take mask as it is generated in masker function.
            masker (callable): Extracts the mask of the finger from the image.
            normalizer (callable): Normalizes the image and mask.
            filter (callable): Applies additional filtering to the image.

        Returns:
            (img, mask): A tuple containing the processed image and mask.
        """
        self._cropped_img = croper(self.image)
        self._mask = masker(self._cropped_img)
        self._norm_img, self._norm_mask = normalizer(self._cropped_img, self._mask)
        self._filtered_img = filter(self._norm_img, self._norm_mask)
        return self._filtered_img, self._norm_mask

    def show(self) -> None:
        """Display a series of image transformations in a 2x3 grid using Matplotlib."""
        _, axes = plt.subplots(2, 3, figsize=(10, 5))

        axes[0][0].imshow(self.image, cmap='gray')
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

# Example preprocessing functions
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

# Example vein extraction functions
def extract_rtl():
    """
    Repeated line tracking algorithm
    https://www.idiap.ch/software/bob/docs/bob/docs/stable/bob/bob.bio.vein/doc/api.html#bob.bio.vein.extractor.RepeatedLineTracking
    """
    # lower number of iterations because 3000 produce thick lines
    return be.RepeatedLineTracking(iterations=600, rescale=False)

def extract_mc():
    """
    Maximum curvature algorithm
    https://www.idiap.ch/software/bob/docs/bob/docs/stable/bob/bob.bio.vein/doc/api.html#bob.bio.vein.extractor.MaximumCurvature
    """
    # lower the std. deviation of the Gaussian filter
    # produces more separated dots around the main vein lines
    return be.MaximumCurvature(sigma=3)

def extract_wl():
    """
    Wide line detector algorithm
    https://www.idiap.ch/software/bob/docs/bob/docs/stable/bob/bob.bio.vein/doc/api.html#bob.bio.vein.extractor.WideLineDetector
    """
    # Cant get this to work nicely
    return be.WideLineDetector(threshold=3,g=35,rescale=False)

def extract_pc():
    """
    Principal curvature
    https://www.idiap.ch/software/bob/docs/bob/docs/stable/bob/bob.bio.vein/doc/api.html#bob.bio.vein.extractor.PrincipalCurvature
    """
    return be.PrincipalCurvature(sigma=3, threshold=4)

if __name__ == "__main__":
    # Example preprocessing transformations (uncomment one)
    # T = preprocess_1()
    T = preprocess_2()
    # T = preprocess_3()

    # Vein extraction algorithms
    E = [extract_rtl(), extract_mc(), extract_wl(), extract_pc()]

    image_preprocessor = BobVeinImage()
    vein_extractor = extractor.BVeinExtractor(E)
    database = db.FingerVeinDatabase()

    # Select 3 random images from database
    batch = database.get_random_batch(3)
    imgs = batch["non_target"]

    # Apply preprocessing transformations to each image
    processed = []
    for img in imgs:
        image_preprocessor.load_image(img)
        processed.append(image_preprocessor.preprocess(*T))
        image_preprocessor.show()

    # Extract and display extracted veins for each preprocessed image
    for p in processed:
        vein_extractor.extract(p)
        vein_extractor.show()
