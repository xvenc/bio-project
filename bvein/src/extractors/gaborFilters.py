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
    
    def rotation_correction(self, img, dst, middle_right, middle_left, w1, w2):
        tangent_value = float(middle_right - middle_left) / float(w2 - w1)
        rotation_angle = np.arctan(tangent_value)/math.pi*180
        (h,w) = img.shape
        center = (w // 2,h // 2)
        M = cv2.getRotationMatrix2D(center,rotation_angle,1)#旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
        rotated_dst = cv2.warpAffine(dst,M,(w,h))
        rotated_img = cv2.warpAffine(img,M,(w,h))
        return rotated_dst, rotated_img

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