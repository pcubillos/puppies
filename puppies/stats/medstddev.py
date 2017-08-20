import numpy as np

__all__ = ["medstd"]

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
  >>> import medstddev as m

  >>> data  = np.array([1,3,4,5,6,7,7])
  >>> std, med = m.medstd(data, median=True)
  >>> print(median(data))
  5.0
  >>> print(med)
  5.0
  >>> print(std)
  2.2360679775
  
  >>> # use masks
  >>> data = np.array([1,3,4,5,6,7,7])
  >>> mask = np.array([1,1,1,0,0,0,0])
  >>> std, med = m.medstd(data, mask, median=True)
  >>> print(std)
  1.58113883008
  >>> print(med)
  3.0

  >>> b = np.array([[1, 3, 4,  5, 6,  7, 7],
                    [4, 3, 4, 15, 6, 17, 7], 
                    [9, 8, 7,  6, 5,  4, 3]])  
  >>> data = np.array([b, 1-b, 2+b])
  >>> std, med = m.medstd(data, median=True, axis=2)
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
  >>> print(m.medstd(data))
  2.2360679775
  """
  # default mask
  if mask is None:
    mask = np.ones(np.shape(data), bool)

  # Make copy, use NaNs as flag:
  d = np.asarray(np.copy(data), float)
  d[~mask] = np.nan

  med   = np.nanmedian(d, axis=axis)
  ngood = np.sum(mask, axis=axis)
  std   = np.sqrt(np.nansum((d - np.expand_dims(med,axis))**2, axis=axis) /
                  (ngood - 1.0))

  if median:
    return std, med
  return std
