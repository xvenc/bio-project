import numpy as np
from .template import BVeinPreprocessor
from scipy.ndimage import convolve
from typing import Tuple

class LeeMask(BVeinPreprocessor):
    def __init__(self, filter_height=4, filter_width=40):
        """
        Implement the Lee mask method for finger vein recognition.

        :param filter_height: Height of the [1, -1] filter.
        :description: If you increase the height, the filter will detect larger
            vertical gradients and may capture more prominent features.
            Decreasing the height makes the filter focus on finer, sharper vertical transitions.

        :param filter_width: Width of the [1, -1] filter.
        """
        self.filter_height = filter_height
        self.filter_width = filter_width

    def preprocess(self, img : np.ndarray, _mask : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ _mask is not used but kept for compatibility with the BVeinPreprocessor interface. """
        # If the image is in uint8 format, convert to float64 and scale to range [0.0, 1.0]
        if img.dtype == np.uint8:
            img = img.astype('float64') / 255.0

        # Get image dimensions
        img_h, img_w = img.shape[:2]

        # Determine lower half starting point
        half_img_h = img_h // 2

        # Construct mask for filtering
        mask = np.ones((self.filter_height, self.filter_width), dtype='float64')
        mask[self.filter_height // 2:, :] = -1.0

        # Filter image using mask (replicating the border similar to 'replicate' in MATLAB)
        img_filt = convolve(img, mask, mode='nearest')

        # Upper part of filtered image
        img_filt_up = img_filt[:half_img_h, :]
        y_up = np.argmax(img_filt_up, axis=0)

        # Lower part of filtered image
        img_filt_lo = img_filt[half_img_h:, :]
        y_lo = np.argmin(img_filt_lo, axis=0)

        # Adjust y_lo indices to match the original image coordinates
        y_lo = y_lo + half_img_h

        # Initialize region mask
        finger_mask = np.zeros_like(img, dtype=bool)

        # Fill region between upper and lower edges
        for i in range(img_w):
            finger_mask[y_up[i]:y_lo[i], i] = True

        return (img, finger_mask)

    def identify_significant_height_differences(self, mask: np.ndarray, height_difference_threshold: float) -> list:
        """
        Identify columns with significant height differences based on the top-most border pixels.

        :param mask: np.ndarray of bools representing the mask.
        :param height_difference_threshold: The threshold for significant height difference (as a percentage).
        :return: A list of tuples indicating the indices of columns with significant height differences.
        """
        img_h, img_w = mask.shape
        half_img_h = img_h // 2  # Middle point of the image (integer division)

        # Find the topmost border pixels in the upper half for each column
        top_pixels_upper_half = np.full(img_w, -1, dtype=int)

        for col in range(img_w):
            # Get indices of `True` values in the upper half
            true_indices_upper_half = np.where(mask[:half_img_h, col])[0]

            if len(true_indices_upper_half) > 0:
                top_pixels_upper_half[col] = true_indices_upper_half[0]  # Top-most `True` value

        # List to store columns with significant height differences
        significant_differences = []

        # Iterate through columns to find significant height differences
        for col in range(1, img_w):  # Start from the second column
            if top_pixels_upper_half[col] != -1 and top_pixels_upper_half[col - 1] != -1:
                height_col = top_pixels_upper_half[col]
                height_prev_col = top_pixels_upper_half[col - 1]

                # Calculate the height difference
                height_diff = abs(height_col - height_prev_col)
                max_height = max(height_col, height_prev_col)
                allowed_difference = height_difference_threshold * max_height

                # Check if the height difference exceeds the allowed threshold
                if height_diff > allowed_difference:
                    significant_differences.append((col - 1, col))  # Store the indices of the columns

        return significant_differences

class ModifiedLeeMask(LeeMask):
    def __init__(self, filter_height=4, filter_width=40, mode='mode1', target_avg_height=0.0, batch_size=20, tolerance_percentage=0.15):
        """
        Implement a modified version of the Lee mask method for finger vein recognition.

        :param filter_height: Height of the [1, -1] filter.
        :param filter_width: Width of the [1, -1] filter.
        :param mode: The mode of operation for the modified Lee mask method.
        :param target_avg_height: The target average height for the columns.
        :param batch_size: The number of columns to process in each batch.
        :param tolerance_percentage: The tolerance percentage for the target average height.
        """
        super().__init__(filter_height, filter_width)
        self.mode = mode
        self.target_avg_height = target_avg_height
        self.batch_size = batch_size
        self.tolerance_percentage = tolerance_percentage

    def __call__(self, img : np.ndarray, _mask : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        original_lee_mask = self.preprocess(img, _mask)
        avg_height = self.get_avg_height(original_lee_mask[1])

        if self.mode == 'mode1':
            mask = self.adjust_columns_to_average_height(original_lee_mask[1], avg_height, self.batch_size)
            mask = np.flipud(mask)
            mask = self.adjust_columns_to_average_height(mask, avg_height, self.batch_size)
            return (img, np.flipud(mask))
        elif self.mode == 'mode2':
            return (img, self.adjust_columns_to_average_height_2(original_lee_mask[1], self.target_avg_height, self.batch_size, self.tolerance_percentage))

    def get_avg_height(self, mask: np.ndarray):
        img_h, img_w = mask.shape
        half_img_h = img_h // 2  # Middle point of the image (integer division)

        # Arrays to store top border pixel positions in the upper half
        top_pixels_upper_half = np.full(img_w, -1, dtype=int)

        # Iterate over each column to find top-most `True` values in the upper half
        for col in range(img_w):
            true_indices_upper_half = np.where(mask[:half_img_h, col])[0]  # Top half only

            if len(true_indices_upper_half) > 0:
                top_pixels_upper_half[col] = true_indices_upper_half[0]  # Top-most `True` value in the upper half

        # Compute average position for top pixels in the upper half
        avg_top_upper_half = np.mean(top_pixels_upper_half[top_pixels_upper_half != -1])  # Ignore columns with no `True` values

        return int(avg_top_upper_half)

    def adjust_columns_to_average_height(self, mask: np.ndarray, target_avg_height: float, batch_size=20) -> np.ndarray:
        """
        Adjust the columns of the mask to match the target average height.

        :param mask: np.ndarray of bools representing the mask.
        :param target_avg_height: The target average height for the columns.
        :param batch_size: The number of columns to process in each batch.
        """
        img_h, img_w = mask.shape
        half_img_h = img_h // 2  # Middle point of the image (integer division)

        # Arrays to store top border pixel positions in the upper half
        top_pixels_upper_half = np.full(img_w, -1, dtype=int)

        # Iterate over each column to find top-most `True` values in the upper half
        for col in range(img_w):
            # Get indices of `True` values in the upper half
            true_indices_upper_half = np.where(mask[:half_img_h, col])[0]

            if len(true_indices_upper_half) > 0:
                top_pixels_upper_half[col] = true_indices_upper_half[0]  # Top-most `True` value in the upper half

        # Adjust columns based on average heights in the upper half
        for start_col in range(0, img_w, batch_size):
            end_col = min(start_col + batch_size, img_w)  # Ensure we don't go out of bounds
            valid_heights = top_pixels_upper_half[start_col:end_col][top_pixels_upper_half[start_col:end_col] != -1]  # Get valid heights

            if len(valid_heights) > 0:
                avg_height = np.mean(valid_heights)  # Compute the average height
                for col in range(start_col, end_col):
                    if top_pixels_upper_half[col] != -1 and top_pixels_upper_half[col] < target_avg_height:
                        # Adjust the height of this column to the target average height
                        current_height = top_pixels_upper_half[col]
                        height_diff = target_avg_height - current_height

                        # Set pixels to True to increase the height
                        for row in range(current_height, min(current_height + height_diff, half_img_h)):
                            mask[row, col] = False

        return mask

    def adjust_columns_to_average_height_2(self, mask: np.ndarray, target_avg_height: float, batch_size=20, tolerance_percentage=0.15) -> np.ndarray:
        """
        Adjust the columns of the mask to match the target average height with a tolerance percentage.

        :param mask: np.ndarray of bools representing the mask.
        :param target_avg_height: The target average height for the columns.
        :param batch_size: The number of columns to process in each batch.
        :param tolerance_percentage: The tolerance percentage for the target average height.
        """
        img_h, img_w = mask.shape
        half_img_h = img_h // 2  # Middle point of the image (integer division)

        # Arrays to store top border pixel positions in the upper half
        top_pixels_upper_half = np.full(img_w, -1, dtype=int)

        # Iterate over each column to find top-most `True` values in the upper half
        for col in range(img_w):
            # Get indices of `True` values in the upper half
            true_indices_upper_half = np.where(mask[:half_img_h, col])[0]

            if len(true_indices_upper_half) > 0:
                top_pixels_upper_half[col] = true_indices_upper_half[0]  # Top-most `True` value in the upper half

        # Define tolerance range around the target average height based on the tolerance percentage
        tolerance = tolerance_percentage * target_avg_height
        lower_bound = target_avg_height - tolerance
        upper_bound = target_avg_height + tolerance

        # Adjust columns based on average heights in the upper half
        for start_col in range(0, img_w, batch_size):
            end_col = min(start_col + batch_size, img_w)  # Ensure we don't go out of bounds
            valid_heights = top_pixels_upper_half[start_col:end_col][top_pixels_upper_half[start_col:end_col] != -1]  # Get valid heights

            if len(valid_heights) > 0:
                avg_height = np.mean(valid_heights)  # Compute the average height
                for col in range(start_col, end_col):
                    current_height = top_pixels_upper_half[col]

                    if current_height != -1:
                        # Check if the current height is outside the tolerance range of the target_avg_height
                        if current_height < lower_bound:
                            # Adjust the height of this column upwards to the lower bound
                            height_diff = lower_bound - current_height
                            for row in range(int(current_height), int(min(current_height + height_diff, half_img_h))):
                                mask[row, col] = False
                        elif current_height > upper_bound:
                            # Adjust the height of this column downwards to the upper bound
                            height_diff = current_height - upper_bound
                            for row in range(int(upper_bound), int(current_height)):
                                mask[row, col] = True

        return mask
