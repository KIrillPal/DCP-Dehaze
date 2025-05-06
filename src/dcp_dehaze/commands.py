from pathlib import Path
import numpy as np
import cv2

from .dcp import DCP
from .filters import (
    guided_filter,
    bilateral_filter,
    median_filter,
    gaussian_filter
)

EPS = 1e-8

def haze(image : np.ndarray, depth : np.ndarray, degree : float = 1.0, min_depth : float = 0.05) -> np.ndarray:
    """
    Apply haze effect to an image based on a depth map.
    
    Args:
        image (np.ndarray): Input image to be hazed.
        depth (np.ndarray): Depth map controlling the haze intensity.
        degree (float, optional): Degree of haze effect. Defaults to 1.0.
    
    Returns:
        np.ndarray: Hazed image with reduced visibility based on depth map.
    """
    
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightest_point = np.unravel_index(np.argmax(grayscale), image.shape[:2])
    
    c_inf = image[brightest_point].astype(np.float64)
    transmission = 1 - np.maximum(depth, min_depth) ** (1 / degree)

    hazed = transmission * image.astype(np.float64) + (1 - transmission) * c_inf
    hazed = np.round(hazed).astype(np.uint8)
    return hazed


def dehaze(
        image : np.ndarray | Path | str, 
        dcp_kernel_type : str | np.ndarray = 'square',
        dcp_kernel_size : int = 15,
        filter_type : str = 'guided',
        filter_size : int = 15,
        t0 : float = 0.1,
        light_quantile : float = 0.999,
        **filter_args
    ) -> np.ndarray:
    """
    Dehaze an image using the Dark Channel Prior (DCP) algorithm.

    Args:
        image (np.ndarray | str | Path): Input image to be dehazed.
        dcp_kernel_type (str | np.ndarray): Type of kernel ('square', 'round') or custom kernel. Defaults to 'square'.
        dcp_kernel_size (int): Size for the DCP kernel. Defaults to 15. 
            If dcp_kernel_type is a custom kernel, this parameter is ignored.
        filter_type (str): Type of filter to use for refining the transmission map. 
            Options: ['guided', 'bilateral', 'median', 'gaussian']. Pass this filter's args to **args.
            See API of the filters in dcp_dehaze.filters
        filter_size (int): Side size for the guided filter. Defaults to 15.
        t0 (float): minimum delimeter for reversing the haze effect. 
            Increasing leads to more stability but less pronounced effect. Defaults to 0.1.
        light_quantile (float): Quantile value for atmospheric light estimation. 
            Higher values lead to more precise but more unstable result. Defaults to 0.999.

    Returns:
        np.ndarray: Dehazed image with reduced visibility.
    """
    # Load the image
    if isinstance(image, Path) or isinstance(image, str):
        image = cv2.imread(str(image))

    # Compute dcp
    if isinstance(dcp_kernel_type, str):
        if dcp_kernel_type == 'square':
            dcp = DCP.compute_square(image, dcp_kernel_size)
        elif dcp_kernel_type == 'round':
            kernel = DCP.round_kernel(dcp_kernel_size)
            dcp = DCP.compute(image, kernel)
        else:
            raise ValueError("dcp_kernel_type must be 'square', 'round' or a custom kernel")
    else:
        kernel = dcp_kernel_type
        dcp = DCP.compute(image, kernel)

    # Enhancing dcp using filter
    if filter_type == 'guided':
        guidance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered_dcp = guided_filter(guidance, dcp, d=filter_size, eps=EPS)
    elif filter_type == 'bilateral':
        filtered_dcp = bilateral_filter(dcp, kernel_size=filter_size, **filter_args)
    elif filter_type == 'median':
        filtered_dcp = median_filter(dcp, kernel_size=filter_size)
    elif filter_type == 'gaussian':
        filtered_dcp = gaussian_filter(dcp, kernel_size=filter_size, **filter_args)
    else:
        raise ValueError("filter_type must be in ['guided', 'bilateral', 'median', 'gaussian']")
    
    # Compute atmospheric light and transmittance map
    c_inf = DCP.compute_atmospheric_light(filtered_dcp, image, light_quantile).astype(np.float64)
    transmission = 1 - filtered_dcp / min(c_inf)

    # Apply per-pixel dehazing
    volume = image.astype(np.float64) - c_inf
    delimeter = np.maximum(transmission, t0)[..., np.newaxis]
    clean_image = volume / delimeter + c_inf
    clean_image = np.clip(clean_image, 0, 255).astype(np.uint8)
    return clean_image