import src.db as db

import bob.bio.vein.preprocessor as bp
import bob.bio.vein.extractor as be

from src.preprocess_wrapper import PreprocessWrapper
from src.extraction_wrapper import ExtractionWrapper

class BobVeinImage(PreprocessWrapper):
    # Overwrite the __init__ method to use bob's preprocessing functions with our image class methods
    def __init__(self, croper, masker, normalizer, filter):
        self.cropper = croper
        self.masker = masker
        self.normalizer = normalizer
        self.filter = filter
        self.process_func = [c for c in [croper, masker, normalizer, filter] if c is not None]

    # We need to overwrite preprocess since bob applies transformations in a specific order
    # and each transformation takes different parameters and returns diffrerent values
    # e.g some require image and mask, some only image, some return only image, some return image and mask
    def apply_preprocessing(self, image_path):
        self._initialize_preprocessing(image_path)

        # Do each function separately and store intermediate results
        img = self.image
        mask = self.mask
        if self.cropper is not None:
            img = self.cropper(img)
            self.process_intermediate.append((img, mask)) # Mask does not change here

        if self.masker is not None:
            mask = self.masker(img)
            self.process_intermediate.append((img, mask)) # Image does not change here

        if self.normalizer is not None:
            img, mask = self.normalizer(img, mask)
            self.process_intermediate.append((img, mask))

        if self.filter is not None:
            img = self.filter(img, mask)
            self.process_intermediate.append((img, mask))

        return img, mask

# Example preprocessing functions
def preprocess_1():
    """
    This preprocessing function uses fixed cropping to remove the top and bottom 30 pixels of the image,
    Lee mask to extract the mask of the finger (I found this mask to work the best), no normalization, and no additional filtering.
    """
    cropper = bp.FixedCrop(top=30, bottom=30)
    masker = bp.LeeMask()
    normalizer = None
    filter = None
    return cropper, masker, normalizer, filter

def preprocess_2():
    """
    Same as preprocess_1 but with histogram equalization to enhance the contrasts of the image.
    """
    cropper = bp.FixedCrop(top=30, bottom=30)
    masker = bp.LeeMask()
    normalizer = None
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

def preprocess_4():
    """ Only LeeMask and HistogramEqualization """
    cropper = None
    masker = bp.LeeMask()
    normalizer = None
    filter = bp.HistogramEqualization()
    return cropper, masker, normalizer, filter

if __name__ == "__main__":
    # Example preprocessing transformations (uncomment one)
    # T = preprocess_1()
    T = preprocess_2()
    # T = preprocess_3()
    # T = preprocess_4()

    # Vein extraction algorithms
    # Descriptions can be found at https://www.idiap.ch/software/bob/docs/bob/docs/stable/bob/bob.bio.vein/doc/api.html
    E = [be.RepeatedLineTracking(iterations=600, rescale=False),
         be.MaximumCurvature(sigma=3),
         be.WideLineDetector(threshold=3,g=35,rescale=False),
         be.PrincipalCurvature(sigma=3, threshold=4)]

    iprep = BobVeinImage(*T)
    extractor = ExtractionWrapper(E)
    database = db.FingerVeinDatabase()

    # Select 3 random images from database
    batch = database.get_random_batch(3)
    imgs = batch["non_target"]

    # Apply preprocessing transformations to each image
    processed = []
    for img in imgs:
        print(f"Processing image: {img}")
        processed.append(iprep.apply_preprocessing(img))
        iprep.show()

    # Extract and display extracted veins for each preprocessed image
    print(f"Extracting veins from image using {[e.__class__.__name__ for e in E]}")
    for img_and_mask in processed:
        extractor.extract(img_and_mask)
        extractor.show()
