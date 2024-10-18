import ctypes, os
import numpy as np
from PIL import Image
from .template import BVeinExtractor
import cv2
from skimage.morphology import skeletonize, binary_dilation, disk

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
    
    def skeletonize(self, img):
        # Convert the image to binary
        _, binary_image = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY)

        # Apply the skeletonize function
        skeleton = skeletonize(binary_image)

        return skeleton

    def dilate(self, img, disk_size=1):
        # Apply binary dilation
        dilated = binary_dilation(img, disk(disk_size))

        return dilated

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
        mean_value = np.mean(vein_array_uint8)

        # Binarize the image using cv2.threshold
        _, binary_image = cv2.threshold(vein_array_uint8, mean_value, 1, cv2.THRESH_BINARY)

        binary_image = binary_image.astype(np.uint8)

        kernel = np.ones((1, 1), np.uint8)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

        #skeleton = self.skeletonize(binary_image)

        #dilated = self.dilate(skeleton, disk_size=1)

        return binary_image 

    def __del__(self):
        # Remove temporary files
        os.remove(self._img_file)
        os.remove(self._mask_file)
        os.remove(self._extracted_file)
