import random, os
from tqdm import tqdm

import bob.bio.vein.algorithm as ba
import bob.bio.vein.preprocessor as bp
import bob.bio.vein.extractor as  be

from bob.bio.vein.preprocessor.utils import draw_mask_over_image
import matplotlib.pyplot as plt

from src.db import FingerVeinDatabase
from src.preprocess.cropper import MaskCropper
from bob_example_methods import BobVeinImage, BobVeinExtractor

from src.match import VeinMatcher

class Comparator():
    def __init__(self, matcher, name):
        self.matcher = matcher
        self.name = name
        self.overall_target_scores = []
        self.overall_non_target_scores = []

    def _compare_veins(self, model, probes):
        """ Compare the model against a list of probe images """
        scores = []
        for imgs in probes:
            score = self.matcher.score(model, imgs)
            scores.append(score)
        return scores

    def compare_target(self, model, probes):
        scores = self._compare_veins(model, probes)
        self.overall_target_scores.extend(scores)

    def compare_non_target(self, model, probes):
        scores = self._compare_veins(model, probes)
        self.overall_non_target_scores.extend(scores)

    def get_scores(self):
        return (self.name, self.overall_target_scores, self.overall_non_target_scores)

def extract_mc():
    return be.MaximumCurvature(sigma=5)

def extract_rtl(iter):
    return be.RepeatedLineTracking(iterations=iter, rescale=False)

def preprocess_pipeline():
    cropper = None
    masker = bp.LeeMask()
    normalizer = None
    filter = bp.HistogramEqualization()
    return cropper, masker, normalizer, filter

def save_extracted_run(exports, name):
    import pickle
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(exports, f)

def load_extracted_run(name):
    import pickle
    with open(f"{name}.pkl", "rb") as f:
        return pickle.load(f)

def print_scores(comp_scores):
    for name, target, non_target in comp_scores:
        target_sorted = sorted(target, reverse=True)
        non_target_sorted = sorted(non_target, reverse=True)
        print(f"{name} - target:")
        for score in target_sorted:
            print(f"\t{score}")
        print(f"{name} - non-target:")
        for score in non_target_sorted:
            print(f"\t{score}")

def save_scores(comp_scores, filename):
    with open(filename, "w") as f:
        for name, target, non_target in comp_scores:
            f.write(f"{name}\n")
            f.write(f"{" ".join(map(str, target))}\n")
            f.write(f"{" ".join(map(str, non_target))}\n")

def extract(db, imgprep, extractor, N, batchsize, filename=None):
    data_to_export = []

    # Get target and non-target images
    for i, batch in enumerate(db.get_random_batch_N(N, batchsize)):
        # Extract the veins from the images
        print(f"Batch {i}")

        # Preprocess the images
        target = [imgprep.apply_preprocessing(img) for img in batch["target"]]
        non_target = [imgprep.apply_preprocessing(img) for img in batch["non_target"]]

        # Extract the veins
        target_ext = [extractor.extract(img, mask)[0] for (img, mask) in tqdm(target)]
        non_target_ext = [extractor.extract(img, mask)[0] for (img, mask) in tqdm(non_target)]

        # Pick a random model from the target images
        single_target_idx = random.randint(0, 5)

        # Save all the data
        data_to_export.append((target, non_target, target_ext, non_target_ext, single_target_idx))

    save_extracted_run(data_to_export, filename)

def evaluate(path):
    exported = load_extracted_run(path)
    comparators = [Comparator(ba.MiuraMatch(), "miura_default"), Comparator(ba.MiuraMatch(cw=30, ch=30), "miura_30"), Comparator(VeinMatcher(), "proposed")]

    maskcropper = MaskCropper()
    for target, _, target_ext, non_target_ext, single_target_idx in exported:
        single_target_image = target_ext[single_target_idx]

        for i in range(len(comparators)):
            if i == len(comparators) - 1:
                # Crop the image by the mask for the proposed method
                single_target_image = maskcropper.preprocess(single_target_image, target[single_target_idx][1])[0]
            comparators[i].compare_target(single_target_image, target_ext)
            comparators[i].compare_non_target(single_target_image, non_target_ext)

    scores = [comp.get_scores() for comp in comparators]
    return scores

if __name__ == '__main__':
    T = preprocess_pipeline()
    E = [extract_rtl(400)]

    imgprep = BobVeinImage(*T)
    extractor = BobVeinExtractor(E)
    db = FingerVeinDatabase()

    random.seed(42)

    eval = False
    eval_dir = "results"

    method_description = "rtl_400"
    runs = 30
    batch_size = 6

    outfile = "_".join([method_description, str(runs), str(batch_size)])

    if eval:
        # Only evaluate, no extraction
        scores = evaluate(outfile)
        print_scores(scores)
        save_scores(scores, os.path.join(eval_dir, outfile))
    else:
        # Only extract, no evaluation
        extract(db, imgprep, extractor, runs, batch_size, outfile)
