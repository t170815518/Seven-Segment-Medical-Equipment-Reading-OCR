import numpy as np
import cv2 as cv
from math import floor, inf
from scipy import ndimage
from scipy.interpolate import interpn


def retinex_filter(img, sampling_spatial, sampling_range, gamma, sigma_range=0.2, sigma_spatial=30, is_show_plot=False):
    """
    Retinex Filter function should be applied to the v channel of HSV picture
    :param img: np.array in 1 channel
    """
    img = np.where(img != 0, img, np.asarray([0.001]))  # avoid 0 value leading to -inf when log
    illumination = np.log(img)
    reflection = illumination

    # find illumination by filtering with envelope mode
    illumination = fast_bilateral_filter(illumination, sigma_spatial, sigma_range, sampling_spatial, sampling_range)

    if is_show_plot:
        cv.imshow("illumination", illumination)
        cv.waitKey(0)

    # find reflection by filtering with regular mode at this point reflection stores original image
    reflection = (reflection - illumination)
    reflection = fast_bilateral_filter(reflection, sigma_spatial, sigma_range, sampling_spatial, sampling_range)

    # apply gamma correction to illumination
    illumination = 1 / gamma * illumination

    return np.exp(reflection + illumination)


def fast_bilateral_filter(img, sigmaSpatial, sigmaRange, samplingSpatial, samplingRange):
    """ Uses convolution with normal gaussian kernel """
    img_size = img.shape

    max_val = np.max(img.flatten())
    min_val = np.min(img.flatten())
    span_val = max_val - min_val

    derivedSigmaSpatial = sigmaSpatial / samplingSpatial
    derivedSigmaRange = sigmaRange / samplingRange

    # to avoid checking points on borders in Gaussian convolution, create a 0 band width
    # and height 'padding' around the grid
    paddingXY = floor(2 * derivedSigmaSpatial) + 1
    paddingZ = floor(2 * derivedSigmaRange) + 1

    # create grid
    downsampledWidth = floor((img_size[1] - 1) / samplingSpatial) + 1 + 2 * paddingXY
    downsampledHeight = floor((img_size[0] - 1) / samplingSpatial) + 1 + 2 * paddingXY
    downsampledDepth = int(floor(span_val / samplingRange) + 1 + 2 * paddingZ)

    grid_wi = np.zeros((downsampledHeight, downsampledWidth, downsampledDepth))
    grid_w = np.zeros((downsampledHeight, downsampledWidth, downsampledDepth))

    # down-sampling
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            pixel_val = img[i, j]

            if not np.isnan(pixel_val):
                x = round(i / samplingSpatial) + paddingXY + 1
                y = round(j / samplingSpatial) + paddingXY + 1
                ksi = int(round((pixel_val - min_val / samplingRange) + paddingZ + 1))

            grid_wi[x, y, ksi] = grid_wi[x, y, ksi] + pixel_val
            grid_w[x, y, ksi] = grid_w[x, y, ksi] + 1

    # create Guassian Kernal
    kernelWidth = 2 * derivedSigmaSpatial + 1
    kernelHeight = kernelWidth
    kernelDepth = 2 * derivedSigmaRange + 1

    grid_wi_b = Gaussian_conv_3d(kernelWidth, kernelHeight, kernelDepth, derivedSigmaSpatial, derivedSigmaRange, grid_wi)
    grid_w_b = Gaussian_conv_3d(kernelWidth, kernelHeight, kernelDepth, derivedSigmaSpatial, derivedSigmaRange, grid_w)

    # divide wi_b / w_b
    normalizedBlurredGrid = np.divide(grid_wi_b, grid_w_b, out=np.zeros_like(grid_wi_b), where=grid_w_b!=0)

    # up-sample
    jj, ii = np.meshgrid(range(img_size[1]), range(img_size[0]))

    di = (ii / samplingSpatial) + paddingXY + 1
    dj = (jj / samplingSpatial) + paddingXY + 1
    dz = (img - min_val) / samplingRange + paddingZ + 1

    # adopt code from matlab: interpn
    pixelGrid = [np.arange(x) for x in normalizedBlurredGrid.shape]
    interpPoints = (np.array([di.flatten(), dj.flatten(), dz.flatten()])).T

    result = interpn(pixelGrid, normalizedBlurredGrid, interpPoints)
    result = np.reshape(result, di.shape)
    return result


def Gaussian_conv_3d(kernelWidth, kernelHeight, kernelDepth, derivedSigmaSpatial, derivedSigmaRange, grid):
    kernalX = gauss2D((kernelWidth, 1), derivedSigmaSpatial)
    kernalY = gauss2D((1, kernelHeight), derivedSigmaSpatial)
    kernalZ = gauss2D((kernelDepth, 1), derivedSigmaRange)
    # expand kernal to be in the same shape as X
    kernalX = np.expand_dims(kernalX, axis=0)
    kernalY = np.expand_dims(kernalY, axis=0)
    kernalZ = np.expand_dims(kernalZ, axis=0)

    k = np.zeros((1, 1, int(round(kernelDepth))))
    k[:] = np.reshape(kernalZ, (1, -1))

    data2 = grid
    data2 = ndimage.convolve(data2, kernalX)  # replace matplotlib.convn
    data2 = ndimage.convolve(data2, kernalY)
    data2 = ndimage.convolve(data2, k)
    return data2


def gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma])
    Ref: https://stackoverflow.com/a/17201686/11180198
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
