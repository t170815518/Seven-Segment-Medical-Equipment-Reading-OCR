import numpy as np
import cv2 as cv
import math
from math import floor, inf
import scipy
from scipy import ndimage, signal
from scipy.interpolate import interpn
import matplotlib.pyplot as plt


def retinex_filter(img, sampling_spatial, sampling_range, gamma, sigma_range=0.2, sigma_spatial=30, is_show_plot=True):
    """
    Retinex Filter function should be applied to the v channel of HSV picture
    :param img: np.array in 1 channel
    """
    img[img == 0] = 0.001  # avoid 0 value leading to -inf when log
    illumination = np.log(img)
    reflection = illumination

    # find illumination by filtering with envelope mode
    illumination = bilateral_approximation(illumination, illumination, sigma_spatial, sigma_range, sampling_spatial, sampling_range)

    # if is_show_plot:
    #     cv.imshow("illumination", reflection - illumination)
    #     cv.waitKey(0)

    # find reflection by filtering with regular mode at this point reflection stores original image
    reflection = (reflection - illumination)
    reflection = bilateral_approximation(reflection, reflection, sigma_spatial, sigma_range, sampling_spatial, sampling_range)

    if is_show_plot:
        plt.imshow(np.exp(illumination), cmap='gray')
        plt.show()
    # apply gamma correction to illumination
    illumination = 1 / gamma * illumination

    return np.exp(reflection + illumination)


def bilateral_approximation(data, edge, sigmaS, sigmaR, samplingS=None, samplingR=None, edgeMin=None, edgeMax=None):
    """ Ref: https://gist.github.com/jackdoerner/b81ad881c4064470d3c0 """
    # This function implements Durand and Dorsey's Signal Processing Bilateral Filter Approximation (2006)
    # It is derived from Jiawen Chen's matlab implementation
    # The original papers and matlab code are available at http://people.csail.mit.edu/sparis/bf/

    inputHeight = data.shape[0]
    inputWidth = data.shape[1]
    samplingS = sigmaS if (samplingS is None) else samplingS
    samplingR = sigmaR if (samplingR is None) else samplingR
    edgeMax = np.amax(edge) if (edgeMax is None) else edgeMax
    edgeMin = np.amin(edge) if (edgeMin is None) else edgeMin
    edgeDelta = edgeMax - edgeMin
    derivedSigmaS = sigmaS / samplingS;
    derivedSigmaR = sigmaR / samplingR;

    paddingXY = math.floor(2 * derivedSigmaS) + 1
    paddingZ = math.floor(2 * derivedSigmaR) + 1

    # allocate 3D grid
    downsampledWidth = math.floor((inputWidth - 1) / samplingS) + 1 + 2 * paddingXY
    downsampledHeight = math.floor((inputHeight - 1) / samplingS) + 1 + 2 * paddingXY
    downsampledDepth = math.floor(edgeDelta / samplingR) + 1 + 2 * paddingZ

    gridData = np.zeros((downsampledHeight, downsampledWidth, downsampledDepth))
    gridWeights = np.zeros((downsampledHeight, downsampledWidth, downsampledDepth))

    # compute downsampled indices
    (jj, ii) = np.meshgrid(range(inputWidth), range(inputHeight))

    di = np.around(ii / samplingS) + paddingXY
    dj = np.around(jj / samplingS) + paddingXY
    dz = np.around((edge - edgeMin) / samplingR) + paddingZ

    # perform scatter (there's probably a faster way than this)
    # normally would do downsampledWeights( di, dj, dk ) = 1, but we have to
    # perform a summation to do box downsampling
    for k in range(dz.size):

        dataZ = data.flat[k]
        if (not math.isnan(dataZ)):
            dik = int(di.flat[k])
            djk = int(dj.flat[k])
            dzk = int(dz.flat[k])

            gridData[dik, djk, dzk] += dataZ
            gridWeights[dik, djk, dzk] += 1

    # make gaussian kernel
    kernelWidth = 2 * derivedSigmaS + 1
    kernelHeight = kernelWidth
    kernelDepth = 2 * derivedSigmaR + 1

    halfKernelWidth = math.floor(kernelWidth / 2)
    halfKernelHeight = math.floor(kernelHeight / 2)
    halfKernelDepth = math.floor(kernelDepth / 2)

    (gridX, gridY, gridZ) = np.meshgrid(range(int(kernelWidth)), range(int(kernelHeight)), range(int(kernelDepth)))
    gridX -= halfKernelWidth
    gridY -= halfKernelHeight
    gridZ -= halfKernelDepth
    gridRSquared = ((gridX * gridX + gridY * gridY) / (derivedSigmaS * derivedSigmaS)) + (
                (gridZ * gridZ) / (derivedSigmaR * derivedSigmaR))
    kernel = np.exp(-0.5 * gridRSquared)

    # convolve
    blurredGridData = signal.fftconvolve(gridData, kernel, mode='same')
    blurredGridWeights = signal.fftconvolve(gridWeights, kernel, mode='same')

    # divide
    blurredGridWeights = np.where(blurredGridWeights == 0, -2,
                                     blurredGridWeights)  # avoid divide by 0, won't read there anyway
    normalizedBlurredGrid = blurredGridData / blurredGridWeights;
    normalizedBlurredGrid = np.where(blurredGridWeights < -1, 0,
                                        normalizedBlurredGrid)  # put 0s where it's undefined

    # upsample
    (jj, ii) = np.meshgrid(range(inputWidth), range(inputHeight))
    # no rounding
    di = (ii / samplingS) + paddingXY
    dj = (jj / samplingS) + paddingXY
    dz = (edge - edgeMin) / samplingR + paddingZ

    return scipy.interpolate.interpn((range(normalizedBlurredGrid.shape[0]), range(normalizedBlurredGrid.shape[1]),
                                      range(normalizedBlurredGrid.shape[2])), normalizedBlurredGrid, (di, dj, dz))
