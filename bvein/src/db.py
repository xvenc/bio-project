import os, random
from typing import List, Dict

class FingerVeinDatabase:
    """
    Database wrapper for the public FingerVein database from kaggle
    available at https://www.kaggle.com/datasets/ryeltsin/finger-vein

    Before using the database make sure that the `BVEIN_DB` environment variable contains
    valid path to the downloaded database root folder. Example: `export BVEIN_DB=/path/to/folder`
    """

    FINGER_TYPES = ['index', 'middle', 'ring']
    HAND_TYPES = ['left', 'right']

    def __init__(self):
        """ Initialize the database with the path to the root folder containing left and right hand images. """
        self.root_folder = self._get_db_path()
        self.database = self._load_database()
        self.batch_used_targets = []

    def _get_db_path(self):
        """ Get database path from environment variable BVEIN_DB. """
        db_path = os.environ.get("BVEIN_DB")
        if not db_path:
            raise ValueError("BVEIN_DB environment variable is not set.")
        return db_path

    def _load_database(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load the folder structure into a dictionary where the key is the subject ID.
        And for each subject, we store hand and for each hand, we store finger and corresponding image paths.
        """
        database = {}
        for subject in os.listdir(self.root_folder):
            subject_path = os.path.join(self.root_folder, subject)
            if not os.path.isdir(subject_path):
                continue
            database[subject] = {x : {} for x in self.HAND_TYPES}
            for hand in self.HAND_TYPES:
                hand_folder = os.path.join(subject_path, hand)
                if not os.path.isdir(hand_folder):
                    continue
                for img_file in os.listdir(hand_folder):
                    finger_name = self._extract_finger_name(img_file)
                    if finger_name is None:
                        continue
                    img_path = os.path.join(hand_folder, img_file)
                    if finger_name not in database[subject][hand]:
                        database[subject][hand][finger_name] = []
                    database[subject][hand][finger_name].append(img_path)

        return database

    def _extract_finger_name(self, image_name: str) -> str:
        """ Extract the finger name from the image file name. """
        if image_name.startswith("index"):
            return "index"
        elif image_name.startswith("middle"):
            return "middle"
        elif image_name.startswith("ring"):
            return "ring"
        return None  # Ignore other files like 'Thumbs.db' or unknown formats

    def get_subjects(self) -> List[str]:
        """ Return a list of hands available in the dataset. """
        return sorted(list(self.database.keys()), key=lambda x: int(x))

    def get_images_by_hand(self, subject : str, hand: str) -> Dict[str, List[str]]:
        """ Return a dictionary of fingers and their corresponding images for a specific hand of subject. """
        return self.database.get(subject, {}).get(hand, {})

    def get_images_by_finger(self, subject: str, hand: str, finger: str) -> List[str]:
        """ Return a list of image paths for a specific finger on a specific hand of subject. """
        return self.database.get(subject, {}).get(hand, {}).get(finger, [])

    def __iter__(self):
        """ Iterate over the database. """
        for subject, hands in self.database.items():
            for hand, fingers in hands.items():
                for finger, images in fingers.items():
                    yield subject, hand, finger, images

    def _generate_random_finger_parameters(self) -> List[str]:
        """ Generate random finger parameters for a batch. """
        subject_idx = random.choice(list(self.database.keys()))
        hand_type = random.choice(self.HAND_TYPES)
        finger_type = random.choice(self.FINGER_TYPES)
        return [subject_idx, hand_type, finger_type]

    def get_random_batch(self, batch_size: int) -> Dict[List[str], List[str]]:
        """
        Get a single batch of randomly selected images.

        Returns:
            batch: A dictionary containing the `target` (6) and `non_target` (`batch_size`) images.
        """
        # Generate new (unused) target
        target = None
        while target is None:
            target_parameters = self._generate_random_finger_parameters()
            if target_parameters not in self.batch_used_targets:
                target = target_parameters
                self.batch_used_targets.append(target_parameters)
        target_imgs = self.get_images_by_finger(*target)

        # Generate non-targets
        non_target_imgs = []
        while batch_size > 0:
            nontarget_parameters = self._generate_random_finger_parameters()
            hand_index = random.randint(0, 5)
            non_target_img = self.get_images_by_finger(*nontarget_parameters)[hand_index]
            if non_target_img not in target and non_target_img not in non_target_imgs:
                non_target_imgs.append(non_target_img)
                batch_size -= 1

        return {"target": target_imgs, "non_target": non_target_imgs}

    def get_random_batch_N(self, N: int, batch_size: int) -> List[Dict[List[str], List[str]]]:
        """ Get `N` batches of randomly selected images. See `get_random_batch` for more details."""
        batches = []
        for _ in range(N):
            batches.append(self.get_random_batch(batch_size))
        return batches

# Example usage
if __name__ == "__main__":
    db = FingerVeinDatabase()

    # List all subjects
    print(", ".join(db.get_subjects()))

    # List paths to all images of subjects 001 left hand
    left_001_all = db.get_images_by_hand('001', 'left')
    for ftype in left_001_all:
        print("\n".join(left_001_all[ftype]))

    # List paths to index finger images of subjects 001 left hand
    print("\n".join(db.get_images_by_finger('001', 'left', 'index')))

    # Iterate through database and display images of hands (only 3 examples)
    imgs_cnt = 0
    for subject, hand, finger, images in db:
        # Images can be iterated here and further processing can be done
        imgs_cnt += len(images)
    print(f"Total images: {imgs_cnt}")

    # Get a single batch of (6) target and 10 non-target images
    random.seed(None) # Don't forget to set the random seed
    batch = db.get_random_batch(10)

    print("Target samples:")
    for img_path in batch["target"]:
        print("\t" + img_path)

    print("Non-target samples:")
    for img_path in batch["non_target"]:
        print("\t" + img_path)
