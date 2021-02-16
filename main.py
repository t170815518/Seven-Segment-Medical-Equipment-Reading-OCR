from PIL import Image
from types import SimpleNamespace
import cv2 as cv
import numpy as np
import random as rng
from math import floor
from skimage.filters import threshold_sauvola, threshold_niblack
from preprocess import preprocess, Sauvola_binarize
from blob_extraction import blob_extraction_CC, blob_extraction_MSER
import matplotlib.pyplot as plt
from homofilt import HomomorphicFilter


if __name__ == '__main__':
    config = SimpleNamespace()
    config.alpha = 0.01
    config.k = 0.2
    config.delta = 0.02
    config.max_are_variant = 0.3


    img_path = "img/bp_test.jpg"
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
    v_binary = v_img
    # v_binary = v_binary.astype(np.uint8)
    # thresh = cv.adaptiveThreshold(v_binary, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (4, 30))
    # dilation = cv.dilate(v_binary, kernel1, iterations=1)
    # erosion = cv.erode(dilation, kernel1, iterations=3)
    # thresh = cv.morphologyEx(v_binary, cv.MORPH_OPEN, kernel1)
    closing = cv.morphologyEx(v_binary, cv.MORPH_OPEN, kernel1)
    plt.imshow(closing, cmap="gray")
    plt.show()
    v_binary = closing
    # blur = cv.GaussianBlur(v_binary, (5, 5), 0)
    v_binary = v_binary.astype(np.uint8)
    # v_binary = cv.adaptiveThreshold(v_binary, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                      cv.THRESH_BINARY, 11, 2)
    _, v_binary = cv.threshold(v_binary, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    plt.imshow(v_binary, cmap="gray")
    plt.show()

    v_binary = v_binary.astype(np.uint8)
    contours, hierarchy = cv.findContours(v_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])

    # Draw polygonal contour + bonding rects + circles
    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv.drawContours(v_binary, contours_poly, i, color)
        cv.rectangle(v_binary, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]),
                                                                                 int(boundRect[i][1]+boundRect[i][3])),
                     color, 2)

    cv.imshow("contours", v_binary)
    cv.waitKey(0)
