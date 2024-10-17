from tqdm import tqdm

from src.db import FingerVeinDatabase
from src.preprocess_wrapper import PreprocessWrapper
from src.extraction_wrapper import ExtractionWrapper

from src.preprocess.lee_mask import ModifiedLeeMask
from src.extractors.RepeatedLineTracking import RepeatedLineTracking

import matplotlib.pyplot as plt

db = FingerVeinDatabase()

preprocessing = [ModifiedLeeMask(mode='mode1')]
imgprep = PreprocessWrapper(preprocessing)

extractor = [RepeatedLineTracking()]
extractors = ExtractionWrapper(extractor)

for i, batch in enumerate(db.get_random_batch_N(1, 2)):
    # Extract the veins from the images
    print(f"Batch {i}")

    # Preprocess the images
    target = [imgprep.apply_preprocessing(img) for img in batch["target"]]
    non_target = [imgprep.apply_preprocessing(img) for img in batch["non_target"]]

    # Extract the veins
    target_ext = [extractors.extract(img_and_mask)[0] for (img_and_mask) in tqdm(target)]
    non_target_ext = [extractors.extract(img_and_mask)[0] for (img_and_mask) in tqdm(non_target)]

    for img in target_ext + non_target_ext:
        plt.imshow(img, cmap="gray")
        plt.show()
