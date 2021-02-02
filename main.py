from PIL import Image
from types import SimpleNamespace
import cv2 as cv
from math import floor
from skimage.filters import threshold_sauvola
from preprocess import preprocess, Sauvola_binarize
from blob_extraction import blob_extraction_CC
import matplotlib.pyplot as plt


if __name__ == '__main__':
    config = SimpleNamespace()
    config.alpha = 0.049
    config.k = 0.15

    img_path = "img/demo.jpg"
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    img = cv.resize(img, (500, 700))
    img_hue = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)  # convert image to hsv
    hue = img_hue[:, :, 0]
    # image enhancement
    v_img = preprocess(img_hue)
    # blob extraction
    window_size = floor(v_img.shape[0] * config.alpha)
    try:
        v_binarised = threshold_sauvola(v_img, window_size, config.k)
    except ValueError:  # when odd is not permitted
        window_size -= 1
        v_binarised = threshold_sauvola(v_img, window_size, config.k)
    blob_extraction_CC(v_binarised, hue)

    plt.imshow(v_binarised, cmap="gray")
    plt.show()
