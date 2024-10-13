import os, random
from src.base import *
import matplotlib.pyplot as plt
import bob.bio.vein.algorithm as ba

FINGER_TYPES = ['index', 'middle', 'ring']
HANDS = ['left', 'right']

def random_finger_parameters():
    """
    Randomly select a finger from the kaggle database
    https://www.kaggle.com/datasets/ryeltsin/finger-vein
    """
    root_idx = str(random.randint(1, 106)).zfill(3)
    hand = random.choice(HANDS)
    ftype = random.choice(FINGER_TYPES)
    fidx = str(random.randint(1, 6))
    return (root_idx, hand, ftype, fidx)

def get_target_imgs(db_path):
    """ Select a random finger from the database and get all its images. """
    ridx, hand, ftype, _ = random_finger_parameters()
    return [os.path.join(db_path, ridx, hand, ftype + "_" + str(i) + '.bmp') for i in range(1, 7)]

def get_nontarget_imgs(db_path, target, N=10):
    """ Select N random fingers from database (that are not the target) and get all their images. """
    imgs = []
    for _ in range(N):
        ridx, hand, ftype, fidx = random_finger_parameters()
        new_imgs_path = os.path.join(db_path, ridx, hand, ftype + "_" + fidx + '.bmp')
        # New image should not be in the target images and should not be in the list of already generated images
        if new_imgs_path not in target and new_imgs_path not in imgs:
            imgs.append(new_imgs_path)
        else:
            N += 1
    return imgs

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
    for img in imgs:
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
    mm = ba.MiuraMatch(ch=40, cw=10)
    scores = []
    for img, data in probes:
        score = mm.score(model, data)
        scores.append(("/".join(img.split('/')[-3:]), score))
    return scores

if __name__ == '__main__':
    procimg = ImagePreprocessor()
    extractor = VeinExtractor()

    # Best preprocessing and extraction methods
    T = custom_preprocess()
    E = [extract_mc()]

    random.seed(1)

    DB_PATH = 'REPLACE_ME' # This should be your path to the database root folder

    # Get target and non-target images
    target = get_target_imgs(DB_PATH)
    non_target = get_nontarget_imgs(DB_PATH, target, N=300)

    # Extract veins from target and non-target images
    target_ext = extract_veins(target)
    non_target_ext = extract_veins(non_target)

    # Pick a random model from the target images
    model = random.choice(target_ext)
    model_name = model[0]
    model_img = model[1]
    print("Model:", model[0])

    # Match the model against all target and non-target images
    target_scores = compare_veins(model_img, target_ext)
    non_target_scores = compare_veins(model_img, non_target_ext)

    # Sort the scores
    target_sorted = sorted(target_scores, key=lambda x: x[1], reverse=True)
    non_target_sorted = sorted(non_target_scores, key=lambda x: x[1], reverse=True)

    print("Target scores:")
    for img, score in target_sorted:
        print(f"{img}: {score}")

    print("Non-target scores:")
    for img, score in non_target_sorted:
        print(f"{img}: {score}")

    # Show extracted veins of 3 lowest scoring target images and 3 highest scoring non-target images
    # show_interesting_veins(model_img, target_sorted[-3:], non_target_sorted[:3])