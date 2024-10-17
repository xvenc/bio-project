import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import det_curve, roc_curve, DetCurveDisplay, RocCurveDisplay

def find_file(phrases):
    # List all files in the current directory
    phrases = phrases.split()
    files = os.listdir()
    for file in files:
        # Check if all phrases are in the file name
        if all(phrase in file for phrase in phrases):
            return file
    return None

def load_data(file_path):
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
            name = lines[0]
            target_scores = [float(value) for value in lines[1].split()]
            non_target_scores = [float(value) for value in lines[2].split()]

            scores[name] = (target_scores, non_target_scores)
    return scores

def roc_and_def(name, target_scores, non_target_scores, axes):
    y_true = [1] * len(target_scores) + [0] * len(non_target_scores)
    y_pred = target_scores + non_target_scores
    fpr, tpr, thr = roc_curve(y_true, y_pred)

    # Display ROC curve
    best = np.argmax(tpr - fpr)
    RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name=name).plot(ax=axes)
    axes.scatter(fpr[best], tpr[best], c='r', s=10)
    print(f"Best threshold for {name}: {thr[best]:.2f}")
    print(f"Best TPR for {name}: {tpr[best]:.2f}")
    print(f"Best FPR for {name}: {fpr[best]:.2f}")

    # Display DET curve
    # fpr, fnr, _ = det_curve(y_true, y_pred)
    # DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name=name).plot(ax=axes[1])

def process_single_file(filename: str):
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    fig.suptitle(datafile)
    scores = load_data(filename)
    for name, (target_scores, non_target_scores) in scores.items():
        roc_and_def(name, target_scores, non_target_scores, axes)

def process_multiple_files(filename: list, matchers: list):
    _, axes = plt.subplots(1, len(matchers), figsize=(14, 6))

    for file in filename:
        scores = load_data(file)
        for matcher_idx in range(len(matchers)):
            target_scores, non_target_scores = scores[matchers[matcher_idx]]
            roc_and_def(file, target_scores, non_target_scores, axes[matcher_idx])
            axes[matcher_idx].set_title(f"{matchers[matcher_idx]} - ROC")

if __name__ == "__main__":
    matchers = ["miura_default", "miura_30", "proposed"]

    # To display ROC and DET for all matchers in a single file
    datafile = find_file("rtl 200")

    # To display ROC and DET for a specific matcher across multiple files
    # datafile = [find_file(x) for x in ["rtl 200", "rtl 400", "rtl 600", "rtl 800"]]
    # datafile = [find_file(x) for x in ["mc5 10", "mc5 6"]]

    if type(datafile) == str:
        process_single_file(datafile)
    else:
        process_multiple_files(datafile, matchers)

    plt.legend(loc='upper right')
    plt.show()