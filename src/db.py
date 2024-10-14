import os
from typing import List, Dict
from PIL import Image

class FingerVeinDatabase:
    def __init__(self, root_folder: str):
        """
        Initialize the database with the path to the root folder containing left and right hand images.
        """
        self.root_folder = root_folder
        self.database = self._load_database()

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
            database[subject] = {"left": {}, "right": {}}
            for hand in ["left", "right"]:
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
        """
        Extract the finger name from the image file name.
        """
        if image_name.startswith("index"):
            return "index"
        elif image_name.startswith("middle"):
            return "middle"
        elif image_name.startswith("ring"):
            return "ring"
        return None  # Ignore other files like 'Thumbs.db' or unknown formats
    
    def get_subjects(self) -> List[str]:
        """
        Return a list of hands available in the dataset.
        """
        return sorted(list(self.database.keys()), key=lambda x: int(x))

    def get_images_by_hand(self, hand: str, subject : str) -> Dict[str, List[str]]:
        """
        Return a dictionary of fingers and their corresponding images for a specific hand of subject.
        """
        return self.database.get(subject, {}).get(hand, {})

    def get_images_by_finger(self, hand: str, finger: str, subject: str) -> List[str]:
        """
        Return a list of image paths for a specific finger on a specific hand of subject.
        """
        return self.database.get(subject, {}).get(hand, {}).get(finger, [])

    def load_image(self, image_path: str) -> Image:
        """
        Load and return an image given its file path.
        """
        return Image.open(image_path)

    def __iter__(self):
        """
        Iterate over the database and yield subject, hand, finger, and images
        """
        for subject, hands in self.database.items():
            for hand, fingers in hands.items():
                for finger, images in fingers.items():
                    yield subject, hand, finger, images

# Example usage:
#db = FingerVeinDatabase("REPLACE_WITH_PATH_TO_DATABASE")
#print(db.get_subjects())
#print(db.get_images_by_hand('left', '001'))
#print(db.get_images_by_finger('left', 'index', '001'))
#for subject, hand, fingers in db:
#    print(f"Subject: {subject}, Hand: {hand}, Fingers: {fingers.keys()}")
#    for finger, images in fingers.items():
#        print(f"\tFinger: {finger}, Images: {images}")
#        for img_path in images:
#            img = db.load_image(img_path)
#            # Further processing of the image can be done here