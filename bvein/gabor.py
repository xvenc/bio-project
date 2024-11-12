from src.db import FingerVeinDatabase
from src.preprocess_wrapper import PreprocessWrapper
from src.extraction_wrapper import ExtractionWrapper

from src.preprocess.lee_mask import ModifiedLeeMask
from src.preprocess.cropper import MaskCropper
from src.preprocess.histogram import HistogramEqualization
from src.extractors.gaborFilters import GaborFilters


preprocessing = [ModifiedLeeMask(), MaskCropper(), HistogramEqualization()]
extractor = [GaborFilters(count=4)]

db = FingerVeinDatabase()
imgprep = PreprocessWrapper(preprocessing)
extractors = ExtractionWrapper(extractor)

# Plot filters
extractors.extractor_funcs[0].plot_filters()

for i, batch in enumerate(db.get_random_batch_N(1, 5)):
    # Extract the veins from the images
    print(f"Batch {i}")

    # Preprocess the images
    target = [imgprep.apply_preprocessing(img) for img in batch["target"]]
    non_target = [imgprep.apply_preprocessing(img) for img in batch["non_target"]]

    # Extract the veins
    for img_and_mask in target + non_target:
        extractors.extract(img_and_mask)
        extractors.extractor_funcs[0].show()
