# Copyright (c) 2021-2024 Patricio Cubillos
# puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

__all__ = [
    'trim',
    'paste',
    ]

import numpy as np


def trim(data, center, size, mask=None, uncert=None, oob=0):
    """
    Extracts a rectangular area of an image masking out of bound pixels.

    Parameters
    ----------
    data: 2D ndarray
        Image from where extract a sub image.
    center: 1D integer ndarray
        y,x position in data where the extracted image will be centered.
    size: 1D integer ndarray
        y,x half-length of the extracted image.  Output image has a
        size of (2*y+1, 2*x+1)
    mask: 2D ndarray
        If specified, this routine will extract the mask subimage
        as well.
    uncert: 2D float ndarray
        If specified, this routine will extract the uncert subimage
        as well.
    oob: scalar
        Value for out of bound pixels in the mask. Default is 0.

    Returns
    -------
    im: 2D float ndarray
        Extracted image.
    mask: 2D integer ndarray
        Extracted mask image.  Out of bound pixels have a value of oob.
    uncert: 2D float ndarray
        Extracted uncert image.  Out of bound pixels have a value of zero.

    Example
    -------
    >>> import imedit as ie
    >>> import numpy as np

    >>> # Create  a data image and its mask
    >>> data  = np.arange(25).reshape(5,5)
    >>> print(data)
    [[ 0  1  2  3  4]
     [ 5  6  7  8  9]
     [10 11 12 13 14]
     [15 16 17 18 19]
     [20 21 22 23 24]]
    >>> msk  = np.ones(np.shape(data))
    >>> msk[1:4,2] = 0

    >>> # Extract a subimage centered on (3,1) of shape (3,5)
    >>> dyc,dxc = 3,1
    >>> subim, mask = ie.trim(data, (dyc,dxc), (1,2), mask=msk)
    >>> print(subim)
    [[ 0 10 11 12 13]
     [ 0 15 16 17 18]
     [ 0 20 21 22 23]]
    >>> print(mask)
    [[0 1 1 0 1]
     [0 1 1 0 1]
     [0 1 1 1 1]]

    >>> # Set out of bound pixels in the mask to -1:
    >>> subim, mask = ie.trim(data, (dyc,dxc), (1,2), mask=msk, oob=-1)
    >>> print(mask)
    [[-1  1  1  0  1]
     [-1  1  1  0  1]
     [-1  1  1  1  1]]
    """
    # Shape of original data
    ny, nx = np.shape(data)

    yc, xc = center
    yr, xr = size
    # The extracted image and mask
    im = np.zeros((2*yr+1, 2*xr+1), dtype=data.dtype)

    # coordinates of the limits of the extracted image
    uplim = yc + yr + 1  # upper limit
    lolim = yc - yr      # lower limit
    rilim = xc + xr + 1  # right limit
    lelim = xc - xr      # left  limit

    # Ranges (in the original image):
    bot = np.amax((0,  lolim))  # bottom
    top = np.amin((ny, uplim))  # top
    lft = np.amax((0,  lelim))  # left
    rgt = np.amin((nx, rilim))  # right

    # Ranges (in the output image):
    hi = top-lolim
    lo = bot-lolim
    le = lft-lelim
    ri = rgt-lelim

    im[lo:hi, le:ri] = data[bot:top, lft:rgt]
    if mask is None and uncert is None:
        return im

    ret = [im]
    if mask is not None:
        ma = np.zeros((2*yr+1, 2*xr+1), int) + oob
        ma[lo:hi, le:ri] = mask[bot:top, lft:rgt]
        ret.append(ma)
    if uncert is not None:
        un = np.zeros((2*yr+1, 2*xr+1), uncert.dtype)
             # + np.amax(uncert[bot:top,lft:rgt])
        un[lo:hi, le:ri] = uncert[bot:top, lft:rgt]
        ret.append(un)

    return ret


def paste(data, subim, center, scenter=None):
    """
    Inserts the subim array into data, the data coordinates (dyc,dxc)
    will match the subim coordinates (syc,sxc). The arrays can have not
    overlapping pixels.

    Parameters:
    ----------
    data: 2D ndarray
        Image where subim will be inserted.
    subim: 2D ndarray
        Image so be inserted.
    center: 1D integer ndarray
        y,x position in data that will match the y,x scenter position
        of subim.
    scenter: 1D integer ndarray
        y,x position in subim matching y,x position in data.  If not
        specified, scenter will be the center of subim.

    Notes
    -----
        This functions modify the input data array, inserting the subim
        array at the specified location.

    Example:
    -------
    >>> import imedit as ie
    >>> import numpy as np
    >>> # Create an array and a subimage array to past in.
    >>> data  = np.zeros((5,5), int)
    >>> subim = np.ones( (3,3), int)
    >>> subim[1,1] = 2
    >>> print(data)
    [[0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]]
    >>> print(subim)
    [[1 1 1]
     [1 2 1]
     [1 1 1]]

    >>> # Define the matching coordinates
    >>> dyc,dxc = 3,1
    >>> syc,sxc = 1,1
    >>> # Paste subim into data
    >>> ie.paste(data, subim, (dyc,dxc), (syc,sxc))
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [1, 1, 1, 0, 0],
     [1, 2, 1, 0, 0],
     [1, 1, 1, 0, 0]]

    >>> # Paste subim into data without a complete overlap between images
    >>> data = np.zeros((5,5), int)
    >>> dyc,dxc = 2,5
    >>> ie.paste(data, subim, (dyc,dxc), (syc,sxc))
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1],
     [0, 0, 0, 0, 1],
     [0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0]]
    """
    # Shape of the arrays
    dny, dnx = np.shape(data)
    sny, snx = np.shape(subim)

    # Center locations:
    if scenter is None:
        scenter = sny/2, snx/2
    dyc, dxc = center
    syc, sxc = scenter

    # left limits:
    led = dxc - sxc
    if led > dnx:  # the entire subimage is out of bounds
        return data

    les = np.amax([0,-led])  # left lim of subimage
    led = np.amax([0, led])  # left lim of data

    # right limits:
    rid = dxc + snx - sxc
    if rid < 0:    # the entire subimage is out of bounds
        return data

    ris = np.amin([snx, dnx - dxc + sxc])  # right lim of subimage
    rid = np.amin([dnx, rid])              # right lim of data

    # lower limits:
    lod = dyc - syc
    if lod > dny:  # the entire subimage is out of bounds
        return data

    los = np.amax([0,-lod])  # lower lim of subimage
    lod = np.amax([0, lod])  # lower lim of data

    # right limits:
    upd = dyc + sny - syc
    if upd < 0:    # the entire subimage is out of bounds
        return data

    ups = np.amin([sny, dny - dyc + syc])  # right lim of subimage
    upd = np.amin([dny, upd])              # right lim of data

    data[lod:upd, led:rid] = subim[los:ups, les:ris]

    return data
