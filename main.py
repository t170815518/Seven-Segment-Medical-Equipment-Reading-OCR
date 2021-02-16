from PIL import Image
from types import SimpleNamespace
import cv2 as cv
import numpy as np
import random as rng
from math import floor
from skimage.filters import threshold_sauvola
from preprocess import preprocess, Sauvola_binarize
from blob_extraction import blob_extraction_CC, blob_extraction_MSER
import matplotlib.pyplot as plt


if __name__ == '__main__':
    config = SimpleNamespace()
    config.alpha = 0.02
    config.k = 0.04
    config.delta = 0.02
    config.max_are_variant = 0.3


    img_path = "img/actual_user_input.PNG"
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    try:
        img = cv.resize(img, (500, 700))
    except cv.error:  # when the img size < (500, 700)
        img = cv.imread(img_path, cv.IMREAD_COLOR)

    img_hue = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)  # convert image to hsv
    hue = img_hue[:, :, 0]
    # image enhancement
    v_img = preprocess(img_hue)

    plt.imshow(v_img, cmap="gray")
    plt.show()
    # blob extraction
    window_size = floor(v_img.shape[0] * config.alpha)

    try:
        v_binarised = threshold_sauvola(v_img, window_size, config.k)
    except ValueError:  # when odd is not permitted
        window_size -= 1
        v_binarised = threshold_sauvola(v_img, window_size, config.k)

    plt.imshow(v_binarised, cmap="gray")
    plt.show()

    v_binarised = v_binarised.astype(np.uint8)
    # thresh = cv.adaptiveThreshold(v_binarised, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    blur = cv.GaussianBlur(v_binarised, (5, 5), 0)
    ret, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    plt.imshow(thresh, cmap="gray")
    plt.show()
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])

    # Draw polygonal contour + bonding rects + circles
    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv.drawContours(v_binarised, contours_poly, i, color)
        cv.rectangle(v_binarised, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]),
                                                                                 int(boundRect[i][1]+boundRect[i][3])),
                     color, 2)

    cv.imshow("contours", v_binarised)
    cv.waitKey(0)