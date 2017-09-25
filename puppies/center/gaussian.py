import os
import sys
import numpy as np
import scipy.optimize as so

from .. import image as im

topdir = os.path.realpath(os.path.dirname(__file__) + "/../..")
sys.path.append(topdir + "/puppies/lib")
import gauss as g

__all__ = ["gaussian", "guess", "fit"]


def gaussian(size, center, sigma, height=0.0, background=0.0):
  """
  Compute a 2D Gaussian function plus a background.

  Parameters
  ----------
  size: 1D integer tuple/ndarray
     (ny,nx) shape of the output array.
  sigma: 1D float tuple/ndarray or scalar
      (y,x)  width of the Gaussian function.  If sigma is a scalar,
      apply same width to x and y axes.
  center: 1D float tuple/ndarray
      (y,x) center location of the Gaussian function.
  height: float
      The height of the Gaussian at center.  If height is zero,
      set height such that the integral of the Gaussian equals one.
  background: float
      Constant factor to add to the output array.

  Returns
  -------
  gauss: 2D float ndarray
    A 2D Gaussian function give by:
         f(x,y) = 1.0/(2*pi*sigmay*sigmax) *
                  exp(-0.5 * (((y-ycenter)/sigmay)**2 +
                             ((y-ycenter)/sigmay)**2 )) + background

  Examples
  --------
  >>> import matplotlib.pyplot as plt
  >>> import puppies.gaussian as g

  >>> # Make 2D Gaussian (Y, X):
  >>> size   = 50, 80
  >>> center =  30, 60
  >>> sigma  =  5,  6.5
  >>> gauss = g.gaussian(size, center, sigma)
  >>> plt.figure(0)
  >>> plt.clf()
  >>> plt.imshow(gauss, origin="lower left", interpolation="nearest")
  >>> plt.title('2D Gaussian')
  >>> plt.xlabel('X')
  >>> plt.ylabel('Y')

  >>> # Gaussian integrates to 1:
  >>> size   = 100, 100
  >>> center =  50, 50
  >>> sigma  =  10, 10
  >>> gauss = g.gaussian(size, center, sigma)
  >>> print(np.sum(gauss))
  0.999998828706
  """
  # Unpack and cast input arguments:
  ny, nx = np.array(size, dtype=int)
  if np.ndim(center) == 0:
    y0 = x0 = float(center)
  else:
    y0, x0 = np.array(center, dtype=float)
  if np.ndim(sigma) == 0:
    sigmay = sigmax = float(sigma)
  else:
    sigmay, sigmax = np.array(sigma, dtype=float)

  # Evaluate the Gaussian:
  gauss = g.gauss2D(ny, nx, y0, x0, sigmay, sigmax,
                    float(height), float(background))

  return gauss


def guess(data, mask=None, yxguess=None):
  """
  Get a guess of the location, width, and height of a 2D Gaussian
  fit to a data array.

  Parameters
  ----------
  data: 2D float ndarray
     Input image to get 2D Gaussian guess
  mask: 2D float ndarray
     Good-pixel mask of data.
  yxguess: 1D integer tuple/ndarray
     If known, the y,x position of the Gaussian.

  Returns
  -------
  ycenter: integer
    Center of the Gaussian along the first dimension.
  xcenter: integer
    Center of the Gaussian along the second dimension.
  sigmay: float
    Gaussian standard deviation along the first dimension.
  sigmax: float
    Gaussian standard deviation along the second dimension.
  height: float
    Height of the Gaussian at (ycenter,xcenter).
  """
  # Default mask:
  if mask is None:
    mask = np.ones(np.shape(data), bool)

  # Center position guess, looking the max value:
  if yxguess is None:
    ycenter, xcenter = np.unravel_index(np.nanargmax(data*mask), np.shape(data))
  else:
    ycenter = int(np.round(yxguess[0]))
    xcenter = int(np.round(yxguess[1]))

  # Height guess is value at center position:
  height = data[ycenter, xcenter]

  # Get sigma from FWHM across the center:
  ywidth = np.sum((data*mask)[:,xcenter] >= 0.5*height)/2.0
  xwidth = np.sum((data*mask)[ycenter,:] >= 0.5*height)/2.0

  return ycenter, xcenter, ywidth, xwidth, height


def fit(data, background=None, fitbg=False, fitguess=None,
        mask=None, weights=None):
  """
  Fit an 2D Gaussian to the data array.

  Parameters
  ----------
  data: ndarray
      Array giving the values of the function.
  background: Float
      Background level
  fitbg: Bool
      Set to True to fit the background level.
  fitguess: tuple, (width, center, height)
      Initial guess of the Gaussian parameters.  This tuple consists
      of: [y0, x0, ysigma, xsigma, height].
  mask: ndarray
      Good pixel mask of data.
  weights : ndarray
      Fit weights of data pixels, typically: 1/sqrt(variance).

  Returns
  -------
  params: ndarray
     This array contains the best fitting values parameters: width,
     center, height, and if requested, bgpars. with:
        width :  The fitted Gaussian widths in each dimension.
        center : The fitted Gaussian center coordinate in each dimension.
        height : The fitted height.
  err: ndarray
     An array containing the concatenated uncertainties
     corresponding to the values of params.  For example, 2D input
     gives np.array([widthyerr, widthxerr, centeryerr, centerxerr,
     heighterr]).

  Notes
  -----
  If the input does not look anything like a Gaussian, the result
  might not even be the best fit to that.

  Method: First guess the parameters (if no guess is provided), then
  call a Levenberg-Marquardt optimizer to finish the job.

  Examples
  --------
  >>> import matplotlib.pyplot as plt
  >>> import puppies.gaussian as g

  >>> # Fit its own model:
  >>> size   = 20,   20
  >>> center = 12.2, 11.95
  >>> sigma  =  1.2,  1.2
  >>> height = 1000.0
  >>> bg = 10.0
  >>> data = g.gaussian(size, center, sigma, height=height, background=bg)
  >>> g.fit(data, fitbg=True)
  array([   12.2 ,    11.95,     1.2 ,     1.2 ,  1000.  ,    10.  ])
  >>> # Perfect fit, as expected.

  >>> # Noise it up:
  >>> noise = np.random.normal(0.0, np.sqrt(data))
  >>> g.fit(data+noise, fitbg=True)
  array([  12.20463709,   11.97439954,    1.20591695,    1.2049832 ,
          992.31654944,   10.06467464])
  >>> # Still pretty good.
  """
  ny, nx = np.shape(data)

  # Defaults:
  if mask is None:
    mask    = np.ones((ny,nx), dtype=bool)
  if weights is None:
    weights = np.ones((ny,nx), dtype=float)

  # Mask the target:
  #medmask = np.copy(mask)
  #if guess is not None:
  #  center = guess[2], guess[3]
  #medmask *= 1 - im.disk(3, center, (ny,nx))
  # Estimate the median of the image:
  #medbg = np.median(data[np.where(medmask)])

  # Estimate background if needed:
  if background is None:
    background = np.median(data[mask])

  # Arguments for residual function:
  args = [ny, nx, data, mask, weights]

  # get a guess if not provided:
  if fitguess is None:
    params = list(guess(data-background, mask=mask))
  else:
    params = list(fitguess)

  # Fit background level if requested:
  if fitbg:
    params += [background]
  else: # Fit to background-subtracted data:
    args[2] = data - background

  # The fit:
  p, cov, info, mesg, success = so.leastsq(residuals, params, tuple(args),
                                           full_output=True)
  return p


def residuals(params, ny, nx, data, mask, weights):
  """
  Calculates the residuals between data and the Gaussian model
  determined by the rest of the parameters. Used in fitgaussian.

  Parameters
  ----------
  params: 1D float tuple/ndarray
      The Gaussian fitting parameters: [y0, x0, ysigma, xsigma, height].
      Optionally, a sixth value fits the background.
  ny: integer
      size of the first dimension of data.
  nx: integer
      size of the second dimension of data.
  data: 2D float ndarray
      Data to be fitted.
  mask: 2D bool ndarray
      Good pixel mask of data (same shape as data).
  weights: 2D float ndarray
      Fitting weights for the values in data (typically, 1/sqrt(variance)).

  Returns
  -------
  residuals: 1D float ndarray
      The masked weighted residuals between the data and the
      Gaussian model.
  """
  # Make the model:
  model = g.gauss2D(ny, nx, *list(params))

  # Calculate residuals:
  res = (model - data) * weights
  # Return only unmasked values:
  return res[mask]
