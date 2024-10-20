import random, os
from tqdm import tqdm

from bob.bio.vein.extractor.RepeatedLineTracking import RepeatedLineTracking as BobRepeatedLineTracking
import bob.bio.vein.algorithm as ba

from src.db import FingerVeinDatabase
from src.match import VeinMatcher
from src.preprocess_wrapper import PreprocessWrapper
from src.extraction_wrapper import ExtractionWrapper

from src.preprocess.lee_mask import LeeMask
from src.preprocess.cropper import MaskCropper
from src.preprocess.histogram import HistogramEqualization
from src.extractors.RepeatedLineTracking import RepeatedLineTracking

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
    export_data = []

    # Get target and non-target images
    for i, batch in enumerate(db.get_random_batch_N(N, batchsize)):
        # Extract the veins from the images
        print(f"Batch {i}")

        # Preprocess the images
        target = [imgprep.apply_preprocessing(img) for img in batch["target"]]
        non_target = [imgprep.apply_preprocessing(img) for img in batch["non_target"]]

        # Extract the veins using using various extraction methods
        target_ext = [extractor.extract(img_and_mask)[0] for img_and_mask in tqdm(target)]
        non_target_ext = [extractor.extract(img_and_mask)[0] for img_and_mask in tqdm(non_target)]

        # Pick a random model from the target images
        single_target_idx = random.randint(0, 5)
        # Save the picked target preprocessed image and mask (because it is used in MaskCropper)
        single_target_image = target_ext[single_target_idx]
        single_target_mask = target[single_target_idx][1]

        # Save all the data
        export_data.append((target_ext, non_target_ext, single_target_image, single_target_mask))

    save_extracted_run(export_data, filename)

def evaluate(path, comparators):
    exported = load_extracted_run(path)

    maskcropper = MaskCropper()
    for target_ext, non_target_ext, single_target_image, single_target_mask in exported:
        for i in range(len(comparators)):
            if i == len(comparators) - 1:
                # Crop the image by the mask for the proposed method
                single_target_image = maskcropper.preprocess(single_target_image, single_target_mask)[0]
            comparators[i].compare_target(single_target_image, target_ext)
            comparators[i].compare_non_target(single_target_image, non_target_ext)

    scores = [comp.get_scores() for comp in comparators]
    return scores

if __name__ == '__main__':
    random.seed(420)

    eval_dir = "results"
    extract_dir = "models"

    runs = 50
    batch_size = 30
    iterations = 1000
    additional_names = ""

    # Define preprocessing functions
    T = [LeeMask(), HistogramEqualization()]

    # Define vein extraction algorithm - always define single here
    # E = [BobRepeatedLineTracking(iterations=800, rescale=False)]
    E = [RepeatedLineTracking(iterations=iterations)]

    # Define comparators
    # C = [
    #     Comparator(ba.MiuraMatch(), "miura_default"),
    #     Comparator(ba.MiuraMatch(cw=30, ch=30), "miura_30"),
    #     Comparator(VeinMatcher(), "proposed")
    # ]
    C = [
        Comparator(VeinMatcher(), "proposed")
    ]

    imgprep = PreprocessWrapper(T)
    extractor = ExtractionWrapper(E)
    db = FingerVeinDatabase()

    outfile = "_".join(extractor.get_extractor_names() + [str(runs), str(batch_size), str(iterations)])
    if additional_names != "":
        outfile += additional_names

    extract(db, imgprep, extractor, runs, batch_size, os.path.join(extract_dir, outfile))
    scores = evaluate(os.path.join(extract_dir, outfile), C)
    save_scores(scores, os.path.join(eval_dir, outfile))
