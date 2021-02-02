from PIL import Image
import cv2 as cv
from preprocess import preprocess


if __name__ == '__main__':
    img_path = "img/demo.jpg"
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    img = cv.resize(img, (500, 700))
    img_hue = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)  # convert image to hsv
    # image enhancement
    img_processed = preprocess(img_hue)
