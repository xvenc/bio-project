import ctypes, os
import numpy as np
from PIL import Image
from .template import BVeinExtractor
import cv2
from skimage.morphology import remove_small_objects

class RepeatedLineTracking(BVeinExtractor):
    def __init__(self, iterations=1000, r=1, profile_w=21):
        self.iterations = iterations
        self.r = r
        self.profile_w = profile_w

        current_dir = os.path.dirname(__file__)
        tmp_dir = "/tmp"

        # Initialize library
        lib_name = "rlt/librlt.so"
        self.lib = ctypes.cdll.LoadLibrary(os.path.join(current_dir, lib_name))

        # Create names for temporary files - located in /tmp directory
        tmp_img_file = "_bvenv_img.png"
        tmp_mask_file = "_bvenv_mask.png"
        tmp_extracted_file = "_bvenv_out.png"
        self._img_file = os.path.join(tmp_dir, tmp_img_file)
        self._mask_file = os.path.join(tmp_dir, tmp_mask_file)
        self._extracted_file = os.path.join(tmp_dir, tmp_extracted_file)

    def extract(self, image_and_mask):
        image, mask = image_and_mask
        # Save image and mask into temporary files
        img = Image.fromarray(image)
        img = img.convert("L")
        img.save(self._img_file)

        mask = Image.fromarray(mask)
        mask = mask.convert("L")
        mask.save(self._mask_file)

        # Call RLT from C++
        self.lib.RepeatedLineTracking(self._img_file.encode(), self._mask_file.encode(), self._extracted_file.encode(), self.iterations, self.r, self.profile_w)

        # Load and return the produced image
        veins = np.asarray(Image.open(self._extracted_file).convert('L'))
        # Convert the array to uint8 type if it's not already
        vein_array_uint8 = np.clip(veins, 0, 255).astype(np.uint8)

        # Calculate the mean value as the threshold
        median_value = np.median(vein_array_uint8[vein_array_uint8 > 0])

        # Binarize the image using cv2.threshold
        _, binary_image = cv2.threshold(vein_array_uint8, median_value, 1, cv2.THRESH_BINARY)

        kernel = np.ones((2, 2), np.uint8)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        # Remove small objects
        binary_image = remove_small_objects(binary_image.astype(bool), min_size=100, connectivity=2).astype(np.uint8)

        # Apply binary opening
        kernel = np.ones((3, 3), np.uint8)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

        return binary_image

    def __del__(self):
        try:
            # Remove temporary files
            os.remove(self._img_file)
            os.remove(self._mask_file)
            os.remove(self._extracted_file)
        except Exception:
            pass
