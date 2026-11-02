"""Classical baseline: dark, thin structures from a local threshold."""

import numpy as np
from scipy.ndimage import binary_opening, uniform_filter


def predict(image: np.ndarray, window: int = 31, offset: float = 18.0) -> np.ndarray:
    gray = image.astype(np.float32).mean(axis=2)
    return binary_opening(gray < uniform_filter(gray, window) - offset, np.ones((2, 2)))
