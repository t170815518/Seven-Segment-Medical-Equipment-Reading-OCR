import cv2 as cv
import numpy as np
from math import sqrt
from retinex_filter import retinex_filter
from PIL.Image import Image
import skimage


def preprocess(img, is_show_img=False):
    """:param img: hsv img array as in OpenCV"""
    img_v = img[:, :, 2]  # value component
    # Branch 1:
    # bilateral filters
    v_retinex = retinex_filter(img_v, 30, 0.2, 1.5)

    # adaptive histogram equalisation
    clahe = cv.createCLAHE(2.0, (8, 8))
    rescaled_img = ((v_retinex - v_retinex.min()) / v_retinex.max() * 255).astype(np.uint8)
    clahe_img_v = clahe.apply(rescaled_img)

    if is_show_img:
        cv.imshow("AHE", clahe_img_v)
        cv.waitKey(0)

    return clahe_img_v


def Sauvola_binarize(imageIn, windowSize, k):
    """ This code is adopted. Unused because of Inefficiency. """
    imageOut = np.zeros_like(imageIn)
    sizeF = windowSize
    sizeF2 = windowSize // 2  # This Value will be used to neglect the edges

    for i in range(sizeF2, imageIn.shape[0] - sizeF2):  # Two loops to move the window
        for j in range(sizeF2, imageIn.shape[1] - sizeF2):  # Starts from 'sizeF2' to neglect the edges
            mean = 0.0
            total = 0
            for m in range(sizeF):  # Two loops to loop through the window
                for n in range(sizeF):
                    mean += imageIn[i + m - sizeF2, j + n - sizeF2]  # add all the pixel's values of the window
                    total += 1  # Increment total
            mean = mean / total

            stand = 0.0
            total = 0
            for m in range(sizeF):
                for n in range(sizeF):
                    stand += (mean - imageIn[i + m - sizeF2, j + n - sizeF2]) ** 2  # Calculating standard deviation
                    total += 1
            stand = sqrt(stand / total)

            # Formula from "Adaptive document image binarization, Jaako Sauvola"
            thres = mean * (1.0 + k * ((stand / 128.0) - 1.0))  # Now that's our threshold value for that pixel

            if imageIn[i, j] > thres:  # if the pixel's value is > Our threshold then our new pixel = 255
                imageOut[i, j] = 255.0  # Affecting the binary values to our new image
            if imageIn[i, j] <= thres:  # if the pixel's value is =< Our threshold then our new pixel = 0
                imageOut[i, j] = 0.0  # Affecting the binary values to our new image
    return imageOut