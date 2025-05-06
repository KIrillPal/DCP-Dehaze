import cv2
import numpy as np


def guided_filter(I, p, d, eps=1e-8):
    """
    Perform guided filtering.

    Parameters:
    I -- guidance image (should be a grayscale/single channel image)
    p -- input image to be filtered (should be a grayscale/single channel image)
    d -- diameter of the window
    eps -- regularization parameter

    Returns:
    q -- filtered image
    """
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (d, d))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (d, d))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (d, d))
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (d, d))

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    A = cov_Ip / (var_I + eps)
    B = mean_p - A * mean_I
    
    a = cv2.boxFilter(A, cv2.CV_32F, (d, d))
    b = cv2.boxFilter(B, cv2.CV_32F, (d, d))
    q = a * I + b
    
    return q


def bilateral_filter(image, kernel_size, color_sigma=1.5, space_sigma=1.5):
    """
    Perform bilateral filtering.

    Parameters:
    image -- input image to be filtered (should be a grayscale/single channel image)
    kernel_size -- diameter of the pixel neighborhood
    color_sigma -- filter sigma in the color space
    space_sigma -- filter sigma in the coordinate space

    Returns:
    filtered_image -- filtered image
    """
    image = image.astype(np.float32)
    return cv2.bilateralFilter(image, kernel_size, color_sigma, space_sigma)


def gaussian_filter(image, kernel_size, sigma=1.5):
    """
    Apply a Gaussian filter to an image.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def median_filter(image, kernel_size):
    """
    Apply a median filter to an image.
    """
    return cv2.medianBlur(image, kernel_size)