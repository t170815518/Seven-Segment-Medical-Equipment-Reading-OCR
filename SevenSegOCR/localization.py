import numpy as np
from math import sqrt
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans


def localization(bands, window_size=(8, 32)):
    height, width = bands[0].shape
    features = []
    for i in range(0, height, window_size[0]):
        for j in range(0, width, window_size[1]):
            window_mask = np.zeros((height, width))
            window_mask[i: i + window_size[0], j: j + window_size[1]] = 1
            feature = featureize(bands, window_mask)
            features.append(feature)
    features = np.concatenate(features, axis=0)
    features, _ = normalize(features, axis=1)
    # clusters
    kmeans = KMeans(2, random_state=1234)
    predict = kmeans.fit_predict(features)


def featureize(bands, window_mask, k=120):
    features = []
    for band in bands:
        band = band[window_mask]
        bins, edges = np.histogram(band, bins=k)
        arr = np.arange(k)
        arr = np.where(bins > 0, x=arr, y=np.zeros_like(arr))
        std = sqrt(np.sum((arr - np.mean(arr)) ** 2) / (k - 1))
        features.append(std)
    return np.array(features)
