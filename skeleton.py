import numpy as np
import scipy.ndimage.morphology as morpho


def skeleton(image_input, structuring_element):
    result = np.zeros_like(image_input).astype(bool)
    tmp = np.copy(image_input).astype(bool)
    while np.any(tmp):
        opening = morpho.binary_opening(tmp, structure=structuring_element)
        top_hat = np.logical_and(tmp, np.logical_not(opening))
        result = np.logical_or(result, top_hat)
        tmp = morpho.binary_erosion(tmp, structure=structuring_element)

    return result.astype(np.int8)
