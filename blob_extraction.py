from PIL import Image
from math import sqrt, floor
import numpy as np
import cv2 as cv
from retinex_filter import retinex_filter


def blob_extraction_CC(img, hue):
    mser = cv.MSER_create()
    regions, bboxes = mser.detectRegions(img.astype(np.uint8))
    hue_values = []
    for region in regions:  # regions: list of point sets
        hue_values.append(hue[region])


def blob_extraction_MSER(img, up, config):
    sigmaSpatial = config.sigmaSpatial
    sigmaRange = config.sigmaRange
    samplingSpatial = sigmaSpatial
    samplingRange = config.sigmaRange
    gamma = config.gamma
    delta = config.delta
    T = config.T

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    value = hsv_img[2]
    hue = hsv_img[0]

    v_retinex = retinex_filter(value, sigmaSpatial, sigmaRange, samplingSpatial, samplingRange, gamma, 0)

    mser = cv.MSER(_area_threshold=delta*100, _max_variation=T)
    points, bboxes = mser.detectRegions(v_retinex)
    img_mask = np.zeros_like(img)
    img_mask[points] = 1

    # Only take one region from all overlapping regions
