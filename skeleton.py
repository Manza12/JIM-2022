import numpy as np
import scipy.ndimage.morphology as morpho


def skeleton(image_input):
    str_el_opening = np.ones((2, 1), dtype=bool)
    result = np.zeros_like(image_input).astype(bool)
    tmp = np.copy(image_input).astype(bool)
    n = 1
    while np.any(tmp):
        str_el_erosion = np.ones((n, 1), dtype=bool)
        opening = morpho.binary_opening(tmp, structure=str_el_opening)
        top_hat = np.logical_and(tmp, np.logical_not(opening))
        result = np.logical_or(result, top_hat)
        tmp = morpho.binary_erosion(image_input, structure=str_el_erosion)
        n += 1

    return result.astype(np.int8)
