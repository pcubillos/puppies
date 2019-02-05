import sys
import os
import numpy as np
from scipy.ndimage.interpolation import map_coordinates

from . import gaussian as g
from . import col

topdir = os.path.realpath(os.path.dirname(__file__) + "/../..")
import asymmetry as a

__all__ = ["asym"]


def asym(data, yxguess, asym_rad=8, asym_size=5, maxcounts=2,
         method='gauss', resize=1.0, weights=None):
  """
  Calculate the center of an input array by first switching the
  array into asymmetry space and finding the minimum

  This centering function works on the idea that the center will be
  the point of minimum asymmetry. To convert to asymmetry space, the
  asymmetry of a radial profile about a particular pixel is
  calculated according to sum(var(r)*Npts(r)), the sum of the
  variance about a particular radius times the number of points at
  that radius. The outter radius of consideration is set by asym_rad.
  The number of points that are converted to asymmetry space is set
  by asym_size producing a shape asym_size*2+1 by asym_size*2+1. This
  asymmetry space is recalculated and moved succesively until the
  point of minimum asymmetry is in the center of the array or
  maxcounts is reached.  Traditional Gaussian or center of light
  centering is then used in the asymmetry space to find the
  sub-pixel point of minimum asymmetry.

  Parameters
  ----------
  data: 2D array
     This is the data to be worked on, the radius of the array
     should at minimum be 2*asym_size.  Recommended: 3 or 4.
  yxguess: 1D tuple/ndarray
     y,x guess of the center
  asym_rad: Integer
     Span of the radial profile used in the asym calculation.
     See notes.
  asym_size: Integer
     Radius of the asym space that is used to determine the center.
     See notes.
  maxcounts: Integer
     Number of times the routine tries to put the point of minimum
     asymmetry in the center of the array.
  method: String
     Centering method of the asymmetry array.  Select between 'gauss'
     to use Gaussian fitting (recommended), or 'col' for center of
     light.
  resize: Float
     Resizing factor for the asym array before centering. Recommended
     scale factors 5 or below.  Resizing introduces a bit of error in
     the center by itself, but may be compensated by the gains in
     precision.  TEST WITH CARE.  This will slow down the function.
  weights: 2D float ndarray
     the weighting that each point should
     recive. low number is less weight, should be type
     float, if none is given the weights are set to one.

  Returns
  -------
  yx_asym: 1D tuple
     The y,x least-asymmetry sub-pixel position of the array.

  Notes
  -----
     This seems to be a good rule of thumb to avoid code breaks:
     - data.shape > arad + asize
     - arad > asize
     I can't explain it, it just works.

  Raises
  ------
  Possibly an assertaion error, it the size of the radial profile is
  different than a given view of the data.  This is most likely due
  to a boundary issue, i.e., the routine is trying to move out of the
  boundary of the input data.  This can be caused by incorrect sizes
  for asym_rad and asym_size, or becuase the routine is trying to walk
  off the edge of data searching for the point of minimum
  asymmetry.  If this happens, try reducing asym_size or asym_rad.

  Example
  -------
  >>> import puppies.center as c
  >>> import puppies.center.gaussian as g

  >>> # Create image:
  >>> size   = 30, 30
  >>> center =  15.1, 15.45
  >>> sigma  =  1.2, 1.2
  >>> data = g.gaussian(size, center, sigma)

  >>> # Least-asymmetry fit:
  >>> yxguess = 15, 14
  >>> arad   = 7
  >>> asize  = 4
  >>> method = "gauss"
  >>> c.asym(data, yxguess, arad, asize, method=method)
  array([ 15.09980257,  15.44999291])
  """
  # Boolean that determines if there are weights to square the variance,
  # to provide larger contrast when using weights
  w_truth = 1
  # Create the weights array if one is not passed in, and set w_truth to 0
  if weights is None:
    weights = np.ones(data.shape, dtype=float)
    w_truth = 0
  elif weights.dtype != np.dtype('float'):
    # Cast to a float if necessary:
    weights = np.array(weights, dtype=float)

  if data.dtype != 'float64':
    data = data.astype('float64')

  x_guess = int(np.round(yxguess[1]))
  y_guess = int(np.round(yxguess[0]))

  # Data indices:
  yind, xind = np.indices((data.shape))

  # Radial profile indices:
  ryind, rxind = np.indices((asym_rad*2+1, asym_rad*2+1))

  # For the course pixel asym location we will reuse the same radial
  # profile:
  dis = np.sqrt((ryind-asym_rad)**2 + (rxind-asym_rad)**2)

  # Positions to calculate an asymmetry value
  suby = yind[y_guess - asym_size:y_guess + asym_size+1,
              x_guess - asym_size:x_guess + asym_size+1]
  shape_save = suby.shape
  suby = suby.flatten()
  subx = xind[y_guess - asym_size:y_guess + asym_size+1,
              x_guess - asym_size:x_guess + asym_size+1]
  subx = subx.flatten()

  # Range statement, as to not recreate it every loop, same with len
  len_y = len(suby)
  iterator = np.arange(len_y)
  ones_len = np.ones(len_y)
  middle   = int(0.5*(len_y - 1))

  # Number of times that the routine has moved pixel space
  counter = 0

  # lambda function used to generate the views outside of the loop
  view = lambda frame, y, x, rng: frame[y-rng:y+rng+1, x-rng:x+rng+1]

  # Start a while loop, to be broken when the maximum number of steps is
  # reached, or when the minimum asymmetry value is in the center of the array
  while counter <= maxcounts:
    # Generator for the views ahead of time that will be needed
    views    = (view(data,    suby[i], subx[i], asym_rad) for i in iterator)
    # Generator for the view on the weights ahead of time
    lb_views = (view(weights, suby[i], subx[i], asym_rad) for i in iterator)

    # Generator to duplicate the distance array for the map function:
    dis_dup     = (dis     for i in ones_len)
    # Generator duplicate for the state of w_truth
    w_truth_dup = (w_truth for i in ones_len)

    # Compute the asymmetry array:
    asym = np.fromiter(map(a.asymmetry, views, dis_dup, lb_views, w_truth_dup),
                       np.double)

    # Move on if the minimum is in the center of the array:
    if np.argmin(asym) == middle:
      break
    # Else, move the array index locations and iterate counter, the
    # while loop then repeats, delete variable to make the garbage
    # collector work less hard
    else:
      suby    += (suby[asym.argmin()]-y_guess)
      subx    += (subx[asym.argmin()]-x_guess)
      counter += 1

  # Return error code if the function waled more times than allowed,
  # (i.e., not finding a center):
  if counter > maxcounts:
    print("maxcounts reached.")
    return np.array([-1.0, -1.0])

  # Else, find the sub-pixel precision and related options
  # First reshape the array to the saved shape
  asym = np.array(asym).reshape(shape_save)
  # Invert the asym space so the minimum point is now the maximum:
  asym = -1.0*asym

  if resize != 1:
    fact = 1.0/float(resize)
    asym = map_coordinates(asym, np.mgrid[0:asym.shape[1]-1+fact:fact,
                                          0:asym.shape[1]-1+fact:fact])

  # Find the sub pixel position using the given method:
  if method == 'col':
    yxfit = col(asym)
  if method == 'gauss':
    yxfit = g.fit(asym)[0:2]

  return (np.array(yxfit)/resize - asym_size +
          np.array((suby[middle], subx[middle]), dtype=float))
