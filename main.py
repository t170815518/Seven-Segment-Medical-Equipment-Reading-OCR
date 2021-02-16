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
import seaborn as sb
from sklearn.preprocessing import normalize
from collections import defaultdict
from pytesseract import image_to_string
import os


if __name__ == '__main__':
    img_path = "img/actual_user_input.PNG"
    BIN_Y = 70
    BIN_X = 50
    DSIZE = (500, 700)
    os.environ["TESSDATA_PREFIX"] = os.path.join(os.getcwd(), "letsgodigital")

    img = cv.imread(img_path, cv.IMREAD_COLOR)
    img = cv.resize(img, DSIZE)
    img = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    cb_img = img[:, :, 2]
    # create a CLAHE object (Arguments are optional).
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cb_img = clahe.apply(cb_img)
    blur = cv.GaussianBlur(cb_img, (5, 5), 0)
    _, binary_img = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    plt.imshow(binary_img, cmap="gray")
    plt.show()

    kernal = cv.getStructuringElement(cv.MORPH_RECT, (90, 10))
    dilation = cv.dilate(binary_img, kernal, iterations=1)
    plt.imshow(dilation, cmap="gray")
    plt.show()

    contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    show_img = cv.cvtColor(binary_img, cv.COLOR_GRAY2BGR)
    boxes = []
    for c in contours:
        (x, y, w, h) = cv.boundingRect(c)
        boxes.append((x, y, w, h))
        cv.rectangle(show_img, (x, y), (x + w, y + h), (255, 255, 0), 4)

    # cv.imshow("contours", show_img)
    # cv.waitKey(0)
    box_info = {}
    for i, box in enumerate(boxes):
        x, y, w, h = box
        crop_img = binary_img[y:y + h, x:x + w]

        plt.imshow(crop_img, cmap="gray")
        plt.show()
        result = image_to_string(crop_img, lang="letsgodigital", config="--psm 13 -c tessedit_char_whitelist=.0123456789")
        box_info[i] = {"area": w * h, "reading": result}
    boxes_info = list(box_info.items())
    box_info = sorted(boxes_info, key=lambda x: x[1]["area"], reverse=True)
    is_valid_result = False
    try:
        while not is_valid_result:
            item = box_info.pop(0)
            reading = float(item[1]["reading"].strip())
            if 0 < reading <= 25:
                is_valid_result = True
                print(reading)
    except IndexError:
        raise ValueError("No reading recognized.")