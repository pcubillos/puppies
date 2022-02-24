# Copyright (c) 2021 Patricio Cubillos
# puppies is open-source software under the MIT license (see LICENSE)

__all__ = [
    'disk',
]

import sys

import numpy as np

from puppies.tools import ROOT
sys.path.append(f"{ROOT}puppies/lib")
import _disk as d


def disk(radius, center, size, status=False, ndisk=False):
    """
    Python wrapper for disk C-extension routine.
    Compute a disk image of given size, with True/False values for those
    pixels whose center lies closer/farther than radius from center.

    Parameters
    ----------
    radius: float
        1D ndarray Radius of the disk, may be fractional.
    center: 1D float ndarray
        y,x position of the center of the disk.
    size: 1D integer ndarray
        y,x size of the output array.
    status: Bool
        Flag to return the out-of-bounds flag.
    ndisk: Bool
        Flag to return the number of pixels in the disk.

    Returns
    -------
    disk: 2D bool ndarray
        Disk image.
    stat: Bool
        A True/False flag indicating if the disk runs/does not run out
        of bounds.
    ndisk: Integer
        Number of pixels in the disk.

    Examples
    --------
    >>> from puppies import image as pi
    >>> disk = pi.disk(3, [4,4], [8,8])
    >>> disk, stat, ndisk = pi.disk(3.0, [4,4], [8,8], status=True, ndisk=True)
    >>> print(stat)
    >>> disk, stat, ndisk = pi.disk(3.1, [4,4], [8,8], True, True)
    >>> print(stat)
    """
    # Cast inputs to the right data-type:
    disk_im, stat, n = d.disk(
        float(radius), np.asarray(center, float), np.asarray(size, int))
    if not status and not ndisk:
        return disk_im

    ret = [disk_im]
    if status:
        ret.append(bool(stat))
    if ndisk:
        ret.append(n)
    return ret
