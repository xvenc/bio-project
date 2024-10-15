import src.db as db
import src.preprocess_wrapper as pw
import src.extraction_wrapper as ew

import bob.bio.vein.preprocessor as bp
import bob.bio.vein.extractor as be

class BobVeinImage(pw.PreprocessWrapper):
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
        img = mask = None
        if self.cropper is not None:
            img = self.cropper(self.image)
            self.process_intermediate.append((img, self.mask)) # Mask does not change here

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

class BobVeinExtractor(ew.ExtractionWrapper):
    def extract(self, image, mask):
        self.image = image
        self.mask = mask
        self.extracted_veins_imgs = [extractor((image, mask)) for extractor in self.extractor_funcs]
        return self.extracted_veins_imgs

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

    iprep = BobVeinImage(*T)
    extractor = BobVeinExtractor(E)
    database = db.FingerVeinDatabase()

    # Select 3 random images from database
    batch = database.get_random_batch(3)
    imgs = batch["non_target"]

    # Apply preprocessing transformations to each image
    processed = []
    for img in imgs:
        processed.append(iprep.apply_preprocessing(img))
        iprep.show()

    # Extract and display extracted veins for each preprocessed image
    for (img, mask) in processed:
        extractor.extract(img, mask)
        extractor.show()
