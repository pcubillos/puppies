import numpy as np

from .. import image as im
from .  import col
from .  import gaussian   as g
from .  import least_asym as la
#import least_asym  as la
#import psf_fit     as pf # TBD

__all__ = ["center"]

def center(method, data, yxguess, trim, arad=4, asize=3,
           mask=None, uncert=None, fitbg=True, maskstar=True,
           expand=1.0, psf=None, psfctr=None):
  """
  Use the center method to find the center of a star in data, starting
  from position guess.

  Parameters
  ----------
  method: string
     Name of the centering method to use.
  data: 2D ndarray
     Array containing the star image.
  yxguess: 2-element 1D float array
     y, x initial guess position of the target.
  trim: Integer
     Semi-lenght of the box around the target that will be trimmed.
  arad: float
     least asymmetry radius parameter.
  asize: float
     least asymmetry size parameter.
     This seems to be a good rule: trim > arad + asize
                                   arad > asize
  mask: 2D ndarray
     A mask array of bad pixels. Same shape of data.
  uncert: 2D ndarray
     An array containing the uncertainty values of data. Same
     shape of data.

  Returns
  -------
  A y,x tuple (scalars) with the coordinates of center of the target
  in data.

  Example
  -------
  >>> import puppies.center as c
  >>> import puppies.center.gaussian as g
  >>> import puppies.image as im
  >>> # Create image:
  >>> size   = 32, 32
  >>> center =  [16.1, 16.45]
  >>> sigma  =  1.2, 1.2
  >>> height = 1000.0
  >>> data = g.gaussian(size, center, sigma, height)

  >>> # Least-asymmetry fit:
  >>> yxguess = 16, 15
  >>> trim  = 8
  >>> arad  = 4
  >>> asize = 3
  >>> # Noise-less fit:
  >>> method = "lag"
  >>> yxlag, extra = c.center(method, data, yxguess, trim, arad,
                              asize, mask=None, uncert=None, fitbg=True)
           #expand=1.0, psf=None, psfctr=None)
  >>> method = "gauss"
  >>> yxgauss, extra = c.center(method, data, yxguess, trim, arad,
                              asize, mask=None, uncert=None, fitbg=True)
  >>> method = "col"
  >>> yxcol, extra = c.center(method, data, yxguess, trim, arad,
                              asize, mask=None, uncert=None, fitbg=True)
  >>> print("True:       {:.5f}, {:.5f}\nleast asym: {:.5f}, {:.5f}\n"
            "Gauss:      {:.5f}, {:.5f}\nCOL:        {:.5f}, {:.5f}".
            format(*center, *yxlag, *yxgauss, *yxcol))
  True:       16.10000, 16.45000
  least asym: 16.09976, 16.44971
  Gauss:      16.10000, 16.45000
  COL:        16.10000, 16.45000

  >>> # Noise it up:
  >>> noise = np.random.normal(0.0, np.sqrt(data))
  >>> yxlag, extra = c.center(method, data+noise, yxguess, trim, arad,
                              asize, mask=None, uncert=None, fitbg=True)
  >>> method = "gauss"
  >>> yxgauss, extra = c.center(method, data+noise, yxguess, trim, arad,
                              asize, mask=None, uncert=None, fitbg=True)
  >>> method = "col"
  >>> yxcol, extra = c.center(method, data+noise, yxguess, trim, arad,
                              asize, mask=None, uncert=None, fitbg=True)
  >>> print("True:       {:.5f}, {:.5f}\nleast asym: {:.5f}, {:.5f}\n"
            "Gauss:      {:.5f}, {:.5f}\nCOL:        {:.5f}, {:.5f}".
            format(*center, *yxlag, *yxgauss, *yxcol))
  True:       16.10000, 16.45000
  least asym: 16.08420, 16.47131
  Gauss:      16.07385, 16.47924
  COL:        16.08420, 16.47131
  >>> # Note, these values will change due to the random noise
  """

  # Default mask: all good
  if mask is None:
    mask = np.ones(np.shape(data), bool)

  # Default uncertainties: flat image
  if uncert is None:
    uncert = np.ones(np.shape(data))

  # Trim the image if requested
  if trim != 0:
    # Integer part of center
    cen = np.array(np.rint(yxguess), int)
    # half-size
    loc = (trim, trim)
    # Do the trim:
    img, msk, err = im.trim(data, cen, loc, mask=mask, uncert=uncert)
  else:
    cen = np.array([0,0])
    loc = np.rint(yxguess)
    img, msk, err = data, mask, uncert

  # If all data is bad:
  if not np.any(msk):
    raise Exception('Bad Frame Exception!')

  weights = 1.0/np.abs(err)
  extra = []

  # Get the center with one of the methods:
  if   method == 'gauss':
    guess = g.guess(img, yxguess=loc)
    y, x = g.fit(img, fitbg=fitbg, fitguess=guess, mask=msk,
                 weights=weights)[0:2]
  elif method == 'col':
    y, x = col(img)
  elif method == 'lag':
    y, x = la.asym(img, loc, asym_rad=arad, asym_size=asize, method='gauss')
  #elif method == 'lac':   # This does not work well
  #  y, x = la.asym(img, loc, asym_rad=radius, asym_size=size, method='col')
  #elif method in ['bpf', 'ipf']:
  #  y, x, flux, sky = pf.spitzer_fit(img, msk, weights, psf, psfctr, expand,
  #                                   method)
  #  extra = flux, sky

  # Make trimming correction and return
  return ((y, x) + cen - trim), extra
