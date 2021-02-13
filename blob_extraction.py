from PIL import Image
from math import sqrt, floor
import numpy as np
import cv2 as cv
from retinex_filter import retinex_filter
from utils import get_bbox_overlap_ratio


def blob_extraction_CC(img, hue=None, threshold=0.0001):
    """

    :param img: binarized
    :param hue: hue channel of the img
    :param threshold:
    :return:
    """
    img = img.astype(np.uint8)  # convert float array to 8-bit
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(img, threshold, 8, cv.CV_32S)
    pass


def blob_extraction_MSER(img, config):
    mser = cv.MSER_create(_delta=int(config.delta * 100), _max_variation=config.max_are_variant)
    regions, bboxes = mser.detectRegions(img.astype(np.uint8))
    areas = bboxes[:, 2] * bboxes[:, 3]
    # get rid of intersecting regions
    keeped = []
    all_bbox = bboxes.T
    for i, (region, bbox) in enumerate(zip(regions, bboxes)):
        overlap_ratios = get_bbox_overlap_ratio(bbox, all_bbox)
        overlap_ratios[i] = 0
        try:
            indices = np.where(overlap_ratios > 0.8)
            indices = np.argmax(areas[indices])
            keeped.append(indices)
        except ValueError:  # attempt to get argmax of an empty sequence
            keeped.append(i)
    # remove redundant element from keeped
    keeped = set(keeped)
    keeped = list(keeped)
    return (np.array(regions)[keeped], bboxes[keeped])