import bvein.src.image as image
from random import seed
from tqdm import tqdm
from datetime import datetime

import matplotlib.pyplot as plt
import bob.bio.vein.algorithm as ba

from bvein.src.db import FingerVeinDatabase

def custom_preprocess():
    """ Custom preprocessing steps """
    cropper = bp.NoCrop()
    masker = bp.LeeMask()
    normalizer = bp.NoNormalization()
    filter = bp.HistogramEqualization()
    return cropper, masker, normalizer, filter

def extract_veins(imgs):
    """ Extract veins from a list of images """
    extracted = []
    for img in tqdm(imgs):
        procimg.load_image(img)
        processed_data = procimg.preprocess(*T)
        extracted.append((img, extractor.extract(E, processed_data)[0]))
    return extracted

def show_interesting_veins(model, target_subopt, non_target_subopt):
    """ Show the model and some interesting target and non-target images """
    def plot(row, data):
        for i, d in enumerate(data):
            axes[row][i].imshow(d[1], cmap='gray')
            axes[row][i].set_title(d[0].split('/')[-1])

    _, axes = plt.subplots(3, 3, figsize=(8, 6))

    axes[0][1].imshow(model, cmap='gray')
    axes[0][1].set_title("Model")

    plot(1, target_subopt)
    plot(2, non_target_subopt)

    list(map(lambda ax: ax.axis('off'), axes.flatten()))
    plt.show()

def compare_veins(model, probes):
    """ Compare the model against a list of probe images """
    mm = ba.MiuraMatch()
    scores = []
    for img, data in probes:
        score = mm.score(model, data)
        scores.append(("/".join(img.split('/')[-3:]), score))
    return scores

if __name__ == '__main__':
    procimg = image.ImagePreprocessor()
    extractor = image.VeinExtractor()

    # Best preprocessing and extraction methods
    T = custom_preprocess()
    E = [image.extract_mc()]

    seed(datetime.now().timestamp())

    DB_PATH = '/home/kali/shared/veindb' # This should be your path to the database root folder

    db = FingerVeinDatabase(DB_PATH)
    # Get target and non-target images
    # for batch in db.get_random_batch_N(10, 10):


    # model_name = random.choice(target)
    # print("Model:", model_name)

    # # Extract veins from target and non-target images
    # target_ext = extract_veins(target)
    # non_target_ext = extract_veins(non_target)

    # # Pick a random model from the target images
    # model_img = target_ext[target.index(model_name)][1]

    # # Match the model against all target and non-target images
    # target_scores = compare_veins(model_img, target_ext)
    # non_target_scores = compare_veins(model_img, non_target_ext)

    # # Sort the scores
    # target_sorted = sorted(target_scores, key=lambda x: x[1], reverse=True)
    # non_target_sorted = sorted(non_target_scores, key=lambda x: x[1], reverse=True)

    # print("Target scores:")
    # for img, score in target_sorted:
    #     print(f"{img}: {score}")

    # print("Non-target scores:")
    # for img, score in non_target_sorted:
    #     print(f"{img}: {score}")

    # Show extracted veins of 3 lowest scoring target images and 3 highest scoring non-target images
    # show_interesting_veins(model_img, target_sorted[-3:], non_target_sorted[:3])