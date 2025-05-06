import numpy as np
from pathlib import Path
import cv2
from abc import ABC, abstractmethod
from scipy.ndimage import minimum_filter

MAX_VALUE = 255

class DCP(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute(image : np.ndarray | Path | str, kernel : np.ndarray) -> np.ndarray:
        """
        Compute the dark channel prior of an image.
        """
        assert kernel.shape[0] == kernel.shape[1], "Kernel must be square"
        assert kernel.shape[0] % 2 == 1, "Kernel must be odd"

        if isinstance(image, Path) or isinstance(image, str):
            image = cv2.imread(str(image))

        height, width = image.shape[:2]
        low_channel = np.min(image, axis=-1).astype(np.float32)
        dcp = np.zeros_like(low_channel)

        d = kernel.shape[0]
        r = d // 2
        low_channel = np.pad(low_channel, r, mode='constant', constant_values=MAX_VALUE)
        
        for i in range(height):
            for j in range(width):
                patch = low_channel[i:i+d, j:j+d]
                dcp[i, j] = np.min(patch + (1 - kernel) * MAX_VALUE)
        
        return dcp.astype(np.uint8)

    @abstractmethod
    def compute_square(image : np.ndarray | Path | str, kernel_size : int) -> np.ndarray:
        """
        Compute the dark channel prior of an image.
        """
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        if isinstance(image, Path) or isinstance(image, str):
            image = cv2.imread(str(image))

        low_channel = np.min(image, axis=-1).astype(np.float32)
        dcp = minimum_filter(low_channel, size=kernel_size, mode='constant', cval=MAX_VALUE)
        return dcp.astype(np.uint8)
    
    @abstractmethod
    def compute_atmospheric_light(
            dcp : np.ndarray, 
            image : np.ndarray, 
            quantile : float = 0.999
        ) -> np.ndarray:
        """
        Compute the atmospheric light of an image.
        """
        background_pixels = np.where(dcp >= np.quantile(dcp, quantile))
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        background_index = np.argmax(grayscale[background_pixels])
        x = background_pixels[0][background_index]
        y = background_pixels[1][background_index]
        return image[x, y]

    @abstractmethod
    def square_kernel(size : int) -> np.ndarray:
        """
        Create a square kernel of the given size.
        """
        assert size % 2 == 1, "Kernel size must be odd"
        return np.ones((size, size))
    
    @abstractmethod
    def round_kernel(size : int) -> np.ndarray:
        """
        Create a round kernel of the given size.
        """
        assert size % 2 == 1, "Kernel size must be odd"
        kernel = np.zeros((size, size))
        r = size // 2
        for i in range(size):
            for j in range(size):
                if (i - r) ** 2 + (j - r) ** 2 <= r ** 2:
                    kernel[i, j] = 1
        return kernel