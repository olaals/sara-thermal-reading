import io

import numpy as np
import tifffile


def load_thermal_tiff(file_path: str) -> np.ndarray:
    """
    Load a thermal TIFF file and return the temperature (Celsius) image as a numpy array.
    """
    image: np.ndarray = tifffile.imread(file_path)
    return image


def load_thermal_tiff_from_bytes(data: bytes) -> np.ndarray:
    """
    Load thermal TIFF data from bytes and return the temperature (Celsius) image as a numpy array.
    """
    image: np.ndarray = tifffile.imread(io.BytesIO(data))
    return image
