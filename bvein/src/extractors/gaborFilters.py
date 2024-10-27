from .template import BVeinExtractor
import cv2
import numpy as np

import matplotlib.pyplot as plt

class GaborFilters(BVeinExtractor):
    def __init__(self, kernel_size=31, sigma=4, lambd=9, gamma=0.5, psi=0, ktype=cv2.CV_32F, count=4):
        """ Initialize the Gabor filter extractor with the given parameters for gabor filters

        Args:
            kernel_size (int): The size of the kernel
            sigma (int): The standard deviation of the gaussian envelope
            lambd (int): The wavelength of the sinusoidal factor
            gamma (float): The spatial aspect ratio
            psi (float): The phase offset
            ktype (int): The type of the filter coefficients
            count (int): The number of filters (and the number of orientations) to build

        """
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.lambd = lambd
        self.gamma = gamma
        self.psi = psi
        self.ktype = ktype

        # Build the filters
        self.filters = self.build_filters(count)

    def build_filters(self, count : int) -> list:
        """ Build a set of gabor filters rotated by theta in the range [0, pi) with a step of pi/count

        Args:
            count (int): The number of filters to build

        Returns:
            filters: A list of gabor filters
        """
        filters = []
        ksize = self.kernel_size
        for theta in np.arange(0, np.pi, np.pi / count):
            kern = cv2.getGaborKernel((ksize, ksize), self.sigma, theta, self.lambd, self.gamma, self.psi, ktype=self.ktype)
            kern /= 1.5*kern.sum()
            filters.append(kern)
        return filters

    def plot_filters(self) -> None:
        # Plot the filters
        fig, axes = plt.subplots(1, len(self.filters))
        fig.suptitle("Filter Bank")
        for i, kern in enumerate(self.filters):
            axes[i].imshow(np.real(kern), cmap="gray")
            axes[i].axis("off")
        plt.tight_layout()
        plt.show()

    def gabor_filter(self, img: np.ndarray) -> np.ndarray:
        """ Apply the gabor filter to the image """
        # Store the filtered images and the accumulative image for later use
        self.fimgs = []
        self.accums = []

        # Accumulative image of the best filter responses
        accum = np.zeros_like(img)
        for kern in self.filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            accum = np.maximum(accum, fimg)
            self.fimgs.append(fimg)
            self.accums.append(accum)

        return accum

    def extract(self, image_and_mask: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """ Extract the veins from the image using the gabor filters

        Args:
            image_and_mask: A tuple containing the image and the mask

        Returns:
            image: The extracted veins from the image
        """
        image, _ = image_and_mask
        self._image_and_mask = image_and_mask

        # Invert the image to make the veins white (for canny edge detection)
        image = np.apply_along_axis(lambda x: 255 - x, 0, image)

        # Apply gaussian blur
        image = cv2.blur(image, (3, 3))

        # Apply gabor filter
        image = self.gabor_filter(image)
        self._ext_veins_img = image

        # Apply canny edge detection
        image = cv2.Canny(image.astype(np.uint8), 90, 150)
        self._ext_canny_img = image

        return image

    def show(self) -> None:
        _, axes = plt.subplots(len(self.fimgs), 2, figsize=(15, 10))

        # Show the filtered images and the accumulative images
        for i, (img, accum) in enumerate(zip(self.fimgs, self.accums)):
            axes[i][0].imshow(img, cmap="gray")
            axes[i][1].imshow(accum, cmap="gray")

        list(map(lambda x: x.axis("off"), axes.flatten()))
        plt.tight_layout()
        plt.show()

        _, axes = plt.subplots(1, 3)

        # Show the original image, the extracted veins, and the canny edge detection
        axes[0].imshow(self._image_and_mask[0], cmap="gray")
        axes[1].imshow(self._ext_veins_img, cmap="gray")
        axes[2].imshow(self._ext_canny_img, cmap="gray")

        list(map(lambda x: x.axis("off"), axes.flatten()))
        plt.tight_layout()
        plt.show()
