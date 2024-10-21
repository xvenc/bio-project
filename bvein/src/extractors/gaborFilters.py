from .template import BVeinExtractor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage import morphology
from skimage.util import invert

class GaborFilters(BVeinExtractor):
    def __init__(self, kernel_size=31, sigma=3.3, lambd=8, gamma=4.5, psi=0.89, ktype=cv2.CV_32F):
    #def __init__(self, kernel_size=11, sigma=9, lambd=8, gamma=1/17, psi=0.89, ktype=cv2.CV_32F):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.lambd = lambd
        self.gamma = gamma
        self.psi = psi
        self.ktype = ktype

    def first_filter(self, image):
        img_blur = cv2.blur(image, (5, 5))
        return img_blur
    
    def edge_detection(self, image):
        sobel_y = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=3)  # Detect horizontal edges

        abs_sobel_y = np.absolute(sobel_y)
        edges = np.uint8(255 * abs_sobel_y / np.max(abs_sobel_y)) 

        return edges

    def pixel_polarization(self, img, img_edges, threshold=25):
        for i in range(len(img_edges)):
            for j in range(len(img_edges[i,:])):
                if img_edges[i,j] > threshold:
                    img[i,j] = 255
                else:
                    img[i,j] = 0
        
        return img

    def positioning_middle_point(self, img, dst, point_pixel):
        h, w = img.shape
        w1 = w // 5  # Left vertical line x-coordinate
        w2 = (w // 5) * 4  # Right vertical line x-coordinate
        
        # Find the middle point on the left side
        low_l, high_l = False, False
        while (not low_l or not high_l) and w1 < (w // 2):
            for i, pix in enumerate(dst[:, w1]):
                if i+1 < (h // 2) and not low_l:
                    if pix == 255:
                        low_l = True
                        lower_left = i
                elif i+1 > (h // 2) and not high_l:
                    h_h = int(h * (3/2) - (i+1))  # Symmetric position search
                    if dst[h_h, w1] == 255:
                        high_l = True
                        higher_left = h_h
            if not low_l or not high_l:
                w1 += 2
        middle_left = (lower_left + higher_left) // 2
        
        # Find the middle point on the right side
        low_r, high_r = False, False
        while (not low_r or not high_r) and w2 > (w // 2):
            for i, pix in enumerate(dst[:, w2]):
                if i+1 < (h // 2) and not low_r:
                    if pix == 255:
                        low_r = True
                        lower_right = i
                elif i+1 > (h // 2) and not high_r:
                    h_h = int(h * (3/2) - (i+1))
                    if dst[h_h, w2] == 255:
                        high_r = True
                        higher_right = h_h
            if not low_r or not high_r:
                w2 -= 2
        middle_right = (lower_right + higher_right) // 2
        
        return dst, middle_left, middle_right, w1, w2

    def rotation_correction(self, img, dst, middle_right, middle_left, w1, w2):
        tangent_value = float(middle_right - middle_left) / float(w2 - w1)
        rotation_angle = np.arctan(tangent_value)/math.pi*180
        (h,w) = img.shape
        center = (w // 2,h // 2)
        M = cv2.getRotationMatrix2D(center,rotation_angle,1)#旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
        rotated_dst = cv2.warpAffine(dst,M,(w,h))
        rotated_img = cv2.warpAffine(img,M,(w,h))
        return rotated_dst, rotated_img

    def roi(self, rotated_img, rotated_edge, w1, w2, margin=25):
        h, w = rotated_edge.shape
        r1 = range(0, h // 2)
        r2 = range(h // 2, h - 1)
        c = range(0, w)
        print(h, w, w1, w2)

        # Find the highest and lowest edges
        highest_edge = (rotated_edge[r1][:,c].sum(axis=1).argmax())
        lowest_edge = (rotated_edge[r2][:,c].sum(axis=1).argmax() + (h // 2))

        # Using w1 and w2 as the leftest and rightest edges
        leftest_edge = max(0, w1 - margin)
        rightest_edge = min(w, w2 + margin) 

        highest_edge = max(0, highest_edge - margin//2)
        lowest_edge = min(h, lowest_edge + margin//2)

        # Cropping the image
        rotated_cropped_img = rotated_img[highest_edge : lowest_edge, leftest_edge : rightest_edge]

        return rotated_cropped_img

    def img_resized_enhance(self, img):
        """Enhances the input image by resizing, normalizing, and applying CLAHE."""
        # Resize the image using bilinear interpolation
        #resized_img = cv2.resize(img, (320, 240), cv2.INTER_LINEAR)

        # Normalize the resized image
        norm_resized_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # Create a CLAHE object and apply it to the normalized image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_resized_img = clahe.apply(norm_resized_img)
        clahe_resized_img = cv2.equalizeHist(clahe_resized_img)

        return clahe_resized_img

    def build_filters(self):
        filters = []
        ksize = self.kernel_size
        for theta in np.arange(0, np.pi, np.pi / 4):
            kern = cv2.getGaborKernel((ksize, ksize), self.sigma, theta, self.lambd, self.gamma, self.psi, ktype=self.ktype)
            kern /= 1.5*kern.sum()
            filters.append(kern)
        return filters

    def gabor_filter(self, img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum

    def extract(self, image_and_mask):
        image, mask = image_and_mask

        img_blur = self.first_filter(image.copy())

        enhanced_img = self.img_resized_enhance(img_blur)

        gab_filters = self.build_filters()
        gabor_image = self.gabor_filter(enhanced_img, gab_filters)

        return gabor_image 

        # Apply the first filter
        #img_blur = self.first_filter(image.copy())

        #edges = self.edge_detection(img_blur)

        #polarity = self.pixel_polarization(image.copy(), edges, threshold=25)

        #img_blur_edge_polar_midd, middle_left, middle_right, w1, w2 = self.positioning_middle_point(image.copy(), polarity, 100)

        #rotated_dst, rotated_img = self.rotation_correction(image.copy(), img_blur_edge_polar_midd, middle_right, middle_left, w1, w2)

        #roi_image = self.roi(rotated_img, rotated_dst, w1, w2)

        #enhanced_img = self.img_resized_enhance(roi_image)

        #gab_filters = self.build_filters()
        #gabor_image = self.gabor_filter(enhanced_img, gab_filters)

        #return gabor_image 