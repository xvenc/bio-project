import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import det_curve, roc_curve, DetCurveDisplay, RocCurveDisplay

def parse_args():
    parser = argparse.ArgumentParser(description="Plot results of the evaluation script stored in results directory")

    # Positional arguments for search phrases
    parser.add_argument('files', nargs='*', help='Files to plot the results from')

    args = parser.parse_args()

    if not args.files:
        parser.error("No files provided")

    return args.files

def load_data(file_path: str) -> dict:
    """ Load the scores from a file.

    The file is read in chunks of three lines. The first line is the name of the matcher,
    the second line is the target scores, and the third line is the non-target scores.
    """
    scores = {}
    try:
        f = open(file_path, 'r')
    except FileNotFoundError:
        print(f"File {file_path} not found")
        exit(1)

    while True:
        # Read three lines from the file
        lines = [f.readline().strip() for _ in range(3)]
        # Filter out empty strings (indicating EOF)
        lines = [line for line in lines if line]
        if not lines:  # If no lines were read, break the loop (EOF reached)
            break
        # Process the three lines
        matcher_name = lines[0]
        target_scores = [float(value) for value in lines[1].split()]
        non_target_scores = [float(value) for value in lines[2].split()]
        scores[matcher_name] = (target_scores, non_target_scores)

    f.close()
    return scores

def roc(name: str, target_scores: list[float], non_target_scores: list[float], axes) -> None:
    """ Calculate and display the ROC curve for the given scores. """
    y_true = [1] * len(target_scores) + [0] * len(non_target_scores)
    y_pred = target_scores + non_target_scores
    fpr, tpr, thr = roc_curve(y_true, y_pred)

    fnr = 1 - tpr
    best = np.nanargmin(np.absolute((fnr - fpr)))
    eer_threshold = thr[best]
    EER = fpr[best]
    print(f"{name}:")
    print(f"\tEER {EER:.2f}")
    print(f"\tEER threshold: {eer_threshold:.2f}")

    # Display ROC curve
    RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name=name).plot(ax=axes)
    axes.scatter(fpr[best], tpr[best], c='black', s=10, zorder=10)

def det(name: str, target_scores: list[float], non_target_scores: list[float], axes) -> None:
    """ Calculate and display the DET curve for the given scores. """
    y_true = [1] * len(target_scores) + [0] * len(non_target_scores)
    y_pred = target_scores + non_target_scores
    fpr, fnr, _ = det_curve(y_true, y_pred)

    DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name=name).plot(ax=axes)
    axes.legend(loc='upper right')

def process_single_file(filename: str, scores: dict, matchers: list) -> None:
    """ Process a single file and display the ROC and DET curves for all matchers found in the file in the same figure. """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(filename)

    for name, (target_scores, non_target_scores) in scores.items():
        roc(name, target_scores, non_target_scores, axes[0])
        det(name, target_scores, non_target_scores, axes[1])

    plt.show()

def process_multiple_files(filenames: list, scores: list, matchers: list):
    """ Process multiple files and display the ROC and DET curves for all matchers found in all files in separate figures. """
    def _plot(filenames, scores, matchers, fn):
        _, axes = plt.subplots(1, len(matchers))
        for i, file in enumerate(filenames):
            file_scores = scores[i]
            file = os.path.basename(file)
            for matcher_idx in range(len(matchers)):
                target_scores, non_target_scores = file_scores[matchers[matcher_idx]]
                if (len(matchers) == 1):
                    fn(file, target_scores, non_target_scores, axes)
                    axes.set_title(f"{matchers[matcher_idx]}")
                else:
                    fn(file, target_scores, non_target_scores, axes[matcher_idx])
                    axes[matcher_idx].set_title(f"{matchers[matcher_idx]}")
        plt.show()

    _plot(filenames, scores, matchers, roc)
    _plot(filenames, scores, matchers, det)

if __name__ == "__main__":
    files = parse_args()

    # Basically there are 2 modes of operation:
    if len(files) == 1:
        # there is a single datafile -> ROC and DET curves for all matchers found in file are plotted in same plot
        filename = files[0]
        scores = load_data(filename)
        matchers_found = list(scores.keys())
        process_single_file(filename, scores, matchers_found)
    else:
        # there are multiple datafiles -> ROC and DET curves are plotted for all matchers found in all files in separate plots
        scores = [load_data(file) for file in files]
        assert all(list(scores[0].keys()) == list(scores[i].keys()) for i in range(1, len(scores))), "All files should have the same matchers"
        matchers_found = list(scores[0].keys())
        process_multiple_files(files, scores, matchers_found)