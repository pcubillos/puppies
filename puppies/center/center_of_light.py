# Copyright (c) 2021 Patricio Cubillos
# puppies is open-source software under the MIT license (see LICENSE)

__all__ = [
    'col',
    ]

import numpy as np


def col(data, weights=None):
    """
    Center of light calculation.

    Parameters
    ----------
    data: 2D float ndarray
        Input image to compute the center of light.
    weights: 2D float ndarray
        Weighting factors for the data pixels.

    Returns
    -------
    yxcol: 2-element float tuple
        The y,x center of mass values.
    """
    if weights is None:
        weights = np.ones(data.shape, dtype=float)

    yind, xind = np.indices(np.shape(data))
    norm = np.sum(weights*data)

    return [
        np.sum(weights*yind*data)/norm,
        np.sum(weights*xind*data)/norm]
