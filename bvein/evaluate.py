import random, os, argparse, pickle
import numpy as np
from tqdm import tqdm

from src.db import FingerVeinDatabase
from src.match import VeinMatcher
from src.preprocess_wrapper import PreprocessWrapper
from src.extraction_wrapper import ExtractionWrapper

from src.preprocess.lee_mask import ModifiedLeeMask
from src.preprocess.cropper import MaskCropper
from src.preprocess.histogram import HistogramEqualization
from src.extractors.RepeatedLineTracking import RepeatedLineTracking

class Comparator():
    """ Class for comparing the extracted veins using a matcher """
    def __init__(self, matcher, name: str) -> None:
        """ Initialize the comparator with a matcher and a name

        Args:
            matcher: The matcher to use for comparing the images - must implement `score` method
            name: The name of the comparator
        """
        self.matcher = matcher
        self.name = name
        self.overall_target_scores = []
        self.overall_non_target_scores = []

    def _compare_veins(self, model: np.ndarray, probes: list[np.ndarray]) -> list[float]:
        """ Compare the model against a list of probe images

        Args:
            model: The model image to compare against
            probes: The list of probe images to compare against

        Returns:
            A list of scores for each probe image
        """
        scores = []
        for imgs in probes:
            score = self.matcher.score(model, imgs)
            scores.append(score)
        return scores

    def compare_target(self, model: np.ndarray, probes: list[np.ndarray]) -> None:
        """ Compare the model against the target images and store the scores

        Args:
            model: The model image to compare against
            probes: The list of target images to compare
        """
        scores = self._compare_veins(model, probes)
        self.overall_target_scores.extend(scores)

    def compare_non_target(self, model: np.ndarray, probes: list[np.ndarray]) -> None:
        """ Compare the model against the non-target images and store the scores

        Args:
            model: The model image to compare against
            probes: The list of non-target images to compare
        """
        scores = self._compare_veins(model, probes)
        self.overall_non_target_scores.extend(scores)

    def get_scores(self) -> tuple[str, list[float], list[float]]:
        """ Get the scores for the comparator - target and non-target """
        return (self.name, self.overall_target_scores, self.overall_non_target_scores)

def save_extracted_run(exports: list, filename: str) -> None:
    """ Save the extracted run (`exports`) to a file using pickle """
    try:
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(exports, f)
    except FileNotFoundError:
        print("The model could not be saved, please check the path")
        exit(1)

def load_extracted_run(filename: str) -> list:
    if ".pkl" not in filename:
        filename += ".pkl"
    try:
        with open(f"{filename}", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("The provided run could not be loaded, please check the path")
        exit(1)

def save_scores(comp_scores: list[str, list[float], list[float]], filename: str) -> None:
    """ Save the scores to a file """
    if ".pkl" in filename:
        filename = filename.replace(".pkl", "")
    with open(filename, "w") as f:
        for name, target, non_target in comp_scores:
            f.write(f"{name}\n")
            f.write(f"{" ".join(map(str, target))}\n")
            f.write(f"{" ".join(map(str, non_target))}\n")

def extract(db: FingerVeinDatabase, imgprep: PreprocessWrapper, extractor: ExtractionWrapper, N: int, batchsize: int, filename: str) -> None:
    """ Extract the veins from the database and save the results to a file

    Args:
        db: The database to extract the images from
        imgprep: The preprocessing wrapper to apply the preprocessing to the images
        extractor: The extraction wrapper to extract the veins from the images
        N: The overall number of runs to perform
        batchsize: The batch size of non-target images used in each run
        filename: The filename to save the results to
    """
    # Since this is usually a long process, check if we can save the results before starting
    if not os.path.exists(os.path.dirname(filename)):
        print("The file does not exist, please check the path")
        exit(1)

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

def evaluate(filename: str, comparators: list[Comparator]) -> list:
    """ Evaluate the extracted veins using the comparators

    Args:
        filename: The filename to load the extracted data from
        comparators: The list of comparators to use for the evaluation

    Returns:
        A list of scores for each comparator
    """
    exported = load_extracted_run(filename)

    maskcropper = MaskCropper()
    for target_ext, non_target_ext, single_target_image, single_target_mask in tqdm(exported):
        for i in range(len(comparators)):
            if comparators[i].name == "proposed":
                # Crop the image by the mask for the proposed method
                single_target_image = maskcropper.preprocess(single_target_image, single_target_mask)[0]
            comparators[i].compare_target(single_target_image, target_ext)
            comparators[i].compare_non_target(single_target_image, non_target_ext)

    scores = [comp.get_scores() for comp in comparators]
    return scores

def parse_args():
    """ Parse the arguments for the script """
    descr = f"""Run evaluation tests on the vein extraction algorithms.
    By default, the script will run the extraction and evaluation in sequence, storing the model in models/ directory and results in results/.
    The script can be run in two modes: full and match. In the full mode, the script will run the extraction and evaluation.
    In the match mode, the script will only run the matchers on the extracted data.
    The file is deduced from the current configuration, but can be overwritten."""
    parser = argparse.ArgumentParser(description=descr)
    # Optional argument for running the matchers on the extracted data
    parser.add_argument('-f', '--file', help='run only the matching methods on provided model stored in file')
    parser.add_argument('-n', help='define number of tests to run', type=int, required=False, default=50)
    parser.add_argument('-b', '--batchsize', help='define batchsize for each test', type=int, required=False, default=30)
    parser.add_argument('-i', '--iterations', help='define number of iterations for RLT', type=int, required=False, default=800)
    args = parser.parse_args()
    assert(args.n > 0 and args.batchsize > 0 and args.iterations > 0)
    return args.file, args.n, args.batchsize, args.iterations

def make_dirs(dirs: list[str]) -> None:
    """ Create directories if they do not exist """
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

if __name__ == '__main__':
    random.seed(None)

    filepath, runs, batch_size, iterations = parse_args()
    match_only = True if filepath else False

    # This will be appended to the model name so you can use it to distinguish between different runs
    additional_names = ""

    # Define preprocessing functions
    T = [ModifiedLeeMask(), HistogramEqualization()]

    # In case of testing Bob RLT, import it here and change the name so its different to our RLT class name

    # from bob.bio.vein.extractor.RepeatedLineTracking import RepeatedLineTracking as BobRLT
    # BobRLT.__name__ = "BobRLT" # Change the name of the class to avoid confusion with our implementation
    # E = [BobRLT(iterations=iterations, rescale=False)]

    # Another example is to use our own implementation
    E = [RepeatedLineTracking(iterations=iterations)]

    # Define comparators - again for the Miura Match we need to import it

    # from bob.bio.vein.algorithm.MiuraMatch import MiuraMatch
    # C = [
    #     Comparator(MiuraMatch(cw = 30, ch=30), "MiuraMatch_cw=ch=30"),
    #     Comparator(MiuraMatch(), "MiuraMatch_default"),
    #     Comparator(VeinMatcher(), "proposed")
    # ]

    # Or just use our own comparator
    C = [
        Comparator(VeinMatcher(), "proposed")
    ]

    # Prepare db, preprocess and extractor
    imgprep = PreprocessWrapper(T)
    extractor = ExtractionWrapper(E)
    db = FingerVeinDatabase()

    eval_dir = "results"
    extract_dir = "models"

    # Generate the filename, based on the configuration, used extractor, N, batchsize and additional names
    if filepath is None:
        filepath = "_".join(extractor.get_extractor_names() + [str(runs), str(batch_size), str(iterations)])
        if additional_names != "":
            filepath += "_" + additional_names
        extraction_file = os.path.join(extract_dir, filepath)
        scores_file = os.path.join(eval_dir, filepath)
    else:
        extraction_file = filepath
        scores_file = os.path.join(eval_dir, os.path.basename(filepath))

    # Make sure the directories exist
    make_dirs([eval_dir, extract_dir])

    # If models have not been provided, run the extraction
    if not match_only:
        print("Running in full mode\nStarting with veins extraction with parameters:")
        print(f"Runs: {runs}\nBatch size: {batch_size}\nIterations: {iterations}")
        print(f"Extracting to {extraction_file}\n")
        extract(db, imgprep, extractor, runs, batch_size, extraction_file)
        print()

    # Run the evaluation
    print("Starting with evaluation")
    print(f"Evaluating {extraction_file}\n")
    scores = evaluate(extraction_file, C)

    # Store the results
    print("\nSaving scores to", os.path.join(eval_dir, os.path.basename(filepath)))
    save_scores(scores, scores_file)
