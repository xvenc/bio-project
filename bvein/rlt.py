import random

from src.db import FingerVeinDatabase
from src.preprocess_wrapper import PreprocessWrapper
from src.extraction_wrapper import ExtractionWrapper

from src.preprocess.cropper import MaskCropper
from src.preprocess.lee_mask import ModifiedLeeMask
from src.preprocess.histogram import HistogramEqualization
from src.extractors.RepeatedLineTracking import RepeatedLineTracking

import matplotlib.pyplot as plt

db = FingerVeinDatabase()
imgprep = PreprocessWrapper([ModifiedLeeMask(), HistogramEqualization(), MaskCropper()])
extractor = ExtractionWrapper([RepeatedLineTracking(iterations=1000)])

random.seed(0)

# Plot the whole pipeline
for batch in db.get_random_batch_N(1, 10):
    for img in batch["target"] + batch["non_target"]:
        _, axes = plt.subplots(1, 4, figsize=(8, 3))

        # Plot original image
        img_original = imgprep._load_image(img)
        axes[0].imshow(img_original, cmap='gray')
        axes[0].set_title("Original")

        # Preprocess the image
        img, mask = imgprep.apply_preprocessing(img)
        axes[1].imshow(img, cmap='gray')
        axes[1].set_title("Preprocessed Image")
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title("Preprocessed Mask")

        img = extractor.extract((img, mask))[0]
        axes[3].imshow(img, cmap='gray')
        axes[3].set_title("Extracted Veins")

        list(map(lambda ax: ax.axis("off"), axes))
        plt.tight_layout()
        plt.show()
