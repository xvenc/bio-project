import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import det_curve, roc_curve, DetCurveDisplay, RocCurveDisplay

RES_DIR = "results"

def find_files(phrases: list[str]) -> list[str]:
    """ List all files in the current directory which contain all the `phrases` in their name. """
    phrases = phrases.split()
    files = os.listdir(RES_DIR)
    files_matched = []
    for file in files:
        # Check if all phrases are in the file name
        if any(phrase in file for phrase in phrases):
            files_matched.append(os.path.join(RES_DIR, file))
    return files_matched

def load_data(file_path: str) -> dict:
    """ Load the scores from a file.

    The file is read in chunks of three lines. The first line is the name of the matcher,
    the second line is the target scores, and the third line is the non-target scores.
    """
    scores = {}
    with open(file_path, 'r') as f:
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
    return scores

def display_shared_legend(fig, matchers_cnt: int) -> None:
    """ Display a shared legend for all subplots in the figure. """
    lines, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', fancybox=True, shadow=True, ncol=matchers_cnt)

def roc(name: str, target_scores: list[float], non_target_scores: list[float], axes) -> None:
    """ Calculate and display the ROC curve for the given scores. """
    y_true = [1] * len(target_scores) + [0] * len(non_target_scores)
    y_pred = target_scores + non_target_scores
    fpr, tpr, thr = roc_curve(y_true, y_pred)

    best = np.argmax(tpr - fpr)
    print(f"Best threshold for {name}: {thr[best]:.2f}")
    print(f"Best TPR for {name}: {tpr[best]:.2f}")
    print(f"Best FPR for {name}: {fpr[best]:.2f}")

    # Display ROC curve
    RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name=name).plot(ax=axes)
    axes.scatter(fpr[best], tpr[best], c='r', s=10)
    axes.grid()
    axes.get_legend().remove()

def det(name: str, target_scores: list[float], non_target_scores: list[float], axes) -> None:
    """ Calculate and display the DET curve for the given scores. """
    y_true = [1] * len(target_scores) + [0] * len(non_target_scores)
    y_pred = target_scores + non_target_scores
    fpr, fnr, _ = det_curve(y_true, y_pred)

    DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name=name).plot(ax=axes)
    axes.grid()
    axes.get_legend().remove()

def process_single_file(scores: dict, matchers: list) -> None:
    """ Process a single file and display the ROC and DET curves for all matchers found in the file in the same figure. """
    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    fig.suptitle(datafile)

    for name, (target_scores, non_target_scores) in scores.items():
        roc(name, target_scores, non_target_scores, axes[0])
        det(name, target_scores, non_target_scores, axes[1])

    display_shared_legend(fig, len(matchers))
    plt.show()

def process_multiple_files(filenames: list, scores: list, matchers: list):
    """ Process multiple files and display the ROC and DET curves for all matchers found in all files in separate figures. """
    def _plot(filenames, scores, matchers, fn):
        fig, axes = plt.subplots(1, len(matchers), figsize=(14, 8))
        for i, file in enumerate(filenames):
            file_scores = scores[i]
            for matcher_idx in range(len(matchers)):
                target_scores, non_target_scores = file_scores[matchers[matcher_idx]]
                if (len(matchers) == 1):
                    fn(file, target_scores, non_target_scores, axes)
                    axes.set_title(f"{matchers[matcher_idx]}")
                else:
                    fn(file, target_scores, non_target_scores, axes[matcher_idx])
                    axes[matcher_idx].set_title(f"{matchers[matcher_idx]}")
        display_shared_legend(fig, len(matchers))
        plt.show()

    _plot(filenames, scores, matchers, roc)
    _plot(filenames, scores, matchers, det)

if __name__ == "__main__":
    datafile = find_files("50 1000")

    # Basically there are 2 modes of operation:
    # 1. Single file mode
    #   - datafile is a string, ROC and DET curves are plotted for all matchers found in file in same plot
    # 2. Multiple files mode
    #   - datafile is a list of strings, ROC and DET curves are plotted for all matchers found in all files in separate plots

    if type(datafile) == str:
        scores = load_data(datafile)
        matchers_found = list(scores.keys())
        process_single_file(scores, matchers_found)
    else:
        scores = [load_data(file) for file in datafile]
        matchers_found = list(scores[0].keys())
        process_multiple_files(datafile, scores, matchers_found)