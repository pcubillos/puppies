# Copyright (c) 2021-2022 Patricio Cubillos
# puppies is open-source software under the MIT license (see LICENSE)

__all__ = [
    'medstd',
    'sigrej',
]

import numpy as np


def medstd(data, mask=None, median=False, axis=0):
    """
    Compute the stddev of an n-dimensional ndarray with
    respect to the median along a given axis.

    Parameters
    ----------
    data: ndarray
        An N-dimensional array from wich to calculate the median standard
        deviation.
    mask: bool ndarray
        Good-value mask.
    median: bool
        If True return a tuple with (stddev, median) of data.
    axis: int
        The axis along wich the median std deviation is calculated.

    Examples
    --------
    >>> import puppies.stats as ps
    >>> import numpy as np
    >>> data = np.array([1,3,4,5,6,7,7])
    >>> std, med = ps.medstd(data, median=True)
    >>> print(median(data))
    5.0
    >>> print(med)
    5.0
    >>> print(std)
    2.2360679775

    >>> # use masks
    >>> data = np.array([1,3,4,5,6,7,7])
    >>> mask = np.array([1,1,1,0,0,0,0], bool)
    >>> std, med = ps.medstd(data, mask, median=True)
    >>> print(std)
    1.58113883008
    >>> print(med)
    3.0

    >>> b = np.array([[1, 3, 4,  5, 6,  7, 7],
                      [4, 3, 4, 15, 6, 17, 7],
                      [9, 8, 7,  6, 5,  4, 3]])
    >>> data = np.array([b, 1-b, 2+b])
    >>> std, med = ps.medstd(data, median=True, axis=2)
    >>> print(median(c, axis=2))
    [[ 5.  6.  6.]
     [-4. -5. -5.]
     [ 7.  8.  8.]]
    >>> print(med)
    [[ 5.  6.  6.]
     [-4. -5. -5.]
     [ 7.  8.  8.]]
    >>> print(std)
    [[ 2.23606798  6.05530071  2.1602469 ]
     [ 2.23606798  6.05530071  2.1602469 ]
     [ 2.23606798  6.05530071  2.1602469 ]]
    >>> # take a look at the first element of std
    >>> data = c[0,0,:]
    >>> print(data)
    [1, 3, 4, 5, 6, 7, 7]
    >>> print(ps.medstd(data))
    2.2360679775
    """
    # default mask
    if mask is None:
        mask = np.ones(np.shape(data), bool)

    # Make copy, use NaNs as flag:
    d = np.asarray(np.copy(data), float)
    d[~mask] = np.nan

    med = np.nanmedian(d, axis=axis)
    ngood = np.sum(mask, axis=axis)
    std = np.sqrt(
        np.nansum((d - np.expand_dims(med,axis))**2, axis=axis) / (ngood - 1.0))

    if median:
        return std, med
    return std


def sigrej(data, sigma, mask=None, axis=0, retival=False,
    retmean=False, retstd=False, retmedian=False, retmedstd=False):
    r"""
    Flag outlying points in a data set using sigma rejection.

    Parameters
    ----------
    data: float ndarray
        Data array where to apply sigma rejection.
    sigma: 1D float ndarray
        Array of sigma values for each iteration of sigma rejection.
        The number of elements determines number of iterations.
    mask: bool ndarray
        Good-pixel mask of data, same shape as data.  Only rejection
        of good-flagged data will be further considered.
    axis: Integer
        Axis along which to evaluate the sigma rejection.
    retival: Bool
        2D array giving the median and standard deviation (with respect
        to the median) at each iteration.
    retmean: Bool
        If True, return the mean of the accepted data.
    retstd:
        If True, return the standard deviation of the accepted data with
        respect to the mean.
    retmedian: Bool
        If True, return the median of the accepted data.
    retmedstd: Bool
        If True, return the standard deviation of the accepted data with
        respect to the median.

    Return
    ------
    This function returns a mask of accepted values in the data.  The
    mask is a byte array of the same shape as Data.  In the mask, 1
    indicates good data, 0 indicates an outlier in the corresponding
    location of Data.

    Notes
    -----
    SIGREJ flags as outliers points a distance of sigma* the standard
    deviation from the median.  The standard deviation is calculated
    with respect to the median, using medstd. For each successive
    iteration and value of sigma, the code recalculates the median and
    standard deviation from the set of 'good' (not masked) values,
    and uses these new values in calculating further outliers.
    The final mask contains a value of True for every 'inlier' and False
    for every outlying data point.

    Example
    -------
    >>> import puppies.stats as ps
    >>> x = np.array([65., 667, 84, 968, 62, 70, 66, 78, 47, 71, 56, 65, 60])

    >>> mask, mean, std, median, medstd = ps.sigrej(
    >>>     x, [2,1], retmean=True, retstd=True, retmedian=True, retmedstd=True)
    >>> print(mask)
    [ True False  True False  True  True  True  True  True  True  True  True
    True]
    >>> print(
    >>>     f"mean,   std:  {mean:.3f}, {std:.3f}\n"
    >>>     f"median, std:  {median:.3f}, {medstd:.3f}")
    mean,   std:  65.818, 10.117
    median, std:  65.000, 10.154
    """
    # Get sizes:
    dims = list(np.shape(data))
    nsig = np.size(sigma)
    if nsig == 0:
        nsig = 1
        sigma = [sigma]

    if mask is None:
        mask = np.ones(dims, bool)

    # Remove axis
    del(dims[axis])
    ival = np.zeros((2, nsig) + tuple(dims))

    for s in range(nsig):
        # Get median and median std:
        ival[1,s], ival[0,s] = medstd(data, mask, axis=axis, median=True)
        # Update mask
        mask *= (
            (data >= ival[0,s] - sigma[s] * ival[1,s]) &
            (data <= ival[0,s] + sigma[s] * ival[1,s])
        )

    # the return arrays
    ret = (mask,)
    if retival:
        ret += (ival,)

    # final calculations
    if retmean or retstd:
        count = np.sum(mask, axis=axis)
        mean = np.nansum(data*mask, axis=axis)

        # calculate only where there are good pixels
        goodvals = np.isfinite(mean) * (count>0)
        if np.ndim(mean) == 0 and goodvals:
            mean /= count
        else:
            mean[np.where(goodvals)] /= count[np.where(goodvals)]

        if retstd:
            resid = (data-mean) * mask
            stddev = np.sqrt(np.sum(resid**2, axis=axis)/(count - 1))
            if np.ndim(stddev) == 0:
                if count == 1:
                    stddev = 0.0
            else:
                stddev[np.where(count == 1)] = 0.0

    # Median stats:
    if retmedian or retmedstd:
        medstddev, median = medstd(data, mask, axis=axis, median=True)

    # the returned final arrays
    if retmean:
        ret += (mean,)
    if retstd:
        ret += (stddev,)
    if retmedian:
        ret += (median,)
    if retmedstd:
        ret += (medstddev,)

    if len(ret) == 1:
        return ret[0]
    return ret
