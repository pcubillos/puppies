# Copyright (c) 2021 Patricio Cubillos
# puppies is open-source software under the MIT license (see LICENSE)

__all__ = [
    'bliss',
    ]

import sys
import numpy as np

from puppies.tools import ROOT
sys.path.append(f"{ROOT}puppies/lib")
import _bilinint as bli


class bliss():
  def __init__(self):
      self.name = "bliss"
      self.type = "pixmap"
      self.pnames = []
      self.npars = len(self.pnames)
      self.params = np.zeros(self.npars)
      self.pmin = np.tile(-np.inf, self.npars)
      self.pmax = np.tile( np.inf, self.npars)
      self.pstep = np.zeros(self.npars)
      self.fixip = False

  def __call__(self, params, model=None):
      """
      Call function with self vaues.
      Update defaults if necessary.
      """
      if model is not None:
          self.model = model
      return self.eval(params, model)


  def eval(self, params, model=None, retmap=False, retbinstd=False):
      """
      Evaluate function at specified input times.
      """
      if model is None:
          model = self.model

      if self.fixip:
          return self.ipflux

      # FINDME: Test calls with retmap, retbinstd
      ipflux = bli.bilinint(
          self.flux, self.model, self.knotpts,
          self.knotsize, self.kploc, self.binloc,
          self.ydist, self.xdist, self.xsize,
          retmap, retbinstd)
      # If requested, put extra variables in self:
      if retmap and retbinstd:
          ipflux, self.blissmap, self.binstd = ipflux
      elif retmap:
          ipflux, self.blissmap = ipflux
      elif retbinstd:
          ipflux, self.binstd   = ipflux

      self.ipflux = ipflux
      return ipflux


  def setup(self, flux=None, y=None, x=None, ystep=None, xstep=None,
      minpt=1, mask=None, verbose=True, obj=None):
      """
      Set up the BLISS map variables.

      Parameters
      ----------
      flux: 1D float ndarray
          Flux values.
      y: 1D float ndarray
          Y-coordinate array of data set (zeroth index).
      x: 1D float ndarray
          X-coordinate array of data set (first index).
      ystep: Float
          Y-coordinate BLISS map grid size.
      xstep: Float
          X-coordinate BLISS map grid size.
      minpt: Integer
          Minimum number of points to accept in a BLISS map tile.
      obj: An object
          [Optional] If not None, extract input arguments from the
          attributes of this object.

      Notes
      -----
      This method defines:
      self.blissmask: 1D integer ndarray
          Mask of accepted (1)/rejected (0) data-set points.
      self.ygrid: 1D float ndarray
          Array of Y-axis coordinates of the knot centers.
      self.xgrid: 1D float ndarray
          Array of X-axis coordinates of the knot centers.
      self.knotpts: 1D integer ndarray
          Data-point indices sorted by knot.
      self.knotsize: 1D integer ndarray
          Number of datapoints per knot.
      self.kploc:  1D integer ndarray
          Index of first data-point index of each knot.
      self.binloc:  1D float ndarray
          Index of the knot to the lower left of the data points.
      self.ydist: 1D float ndarray
          Normalized distance to the bottom knot (binloc).
      self.xdist: 1D float ndarray
          Normalized distance to the left knot (binloc).
      """
      # Retrieve data from pup (obj) object:
      if obj is not None:
          flux = obj.flux
          y = obj.y
          x = obj.x
          ystep = obj.ystep
          xstep = obj.xstep
          minpt = obj.minpt

      if mask is None:
          mask = np.ones(len(flux), bool)

      # Determine the centroid location relative to the center of pixel:
      # Record in which quadrant the center of light falls:
      yround = np.round(np.median(y))
      xround = np.round(np.median(x))
      if verbose:
          print(f"Reference pixel position (y,x): ({yround:.0f}, {xround:.0f})")

      # Put the first knot one step to the right of the left-most point:
      ymin = np.amin(y[mask]) - ystep
      xmin = np.amin(x[mask]) - xstep

      # Calculate number of necessary cells:
      ysize = int((np.amax(y[mask]) - ymin)/ystep + 0.5) + 2
      xsize = int((np.amax(x[mask]) - xmin)/xstep + 0.5) + 2
      self.xsize = xsize

      # Position of the last knot:
      ymax = ymin + (ysize-1)*ystep
      xmax = xmin + (xsize-1)*xstep

      # Make the grid of knot coordinates:
      self.ygrid = np.linspace(ymin, ymax, ysize)
      self.xgrid = np.linspace(xmin, xmax, xsize)

      if verbose:
          print(
              f'BLISS map boundaries: x=[{xmin:.3f}, {xmax:.3f}], '
              f'y=[{ymin:.3f}, {ymax:.3f}]')
          print(f'BLISS map size: {ysize:d} x {xsize:d}')
          print(f'Step size in y = {ystep:.6g}')
          print(f'Step size in x = {xstep:.6g}')
          print(f'Ignoring bins with < {minpt:d} points.')

      # Make mask for minimum number of points:
      self.mask = mask
      ndata = np.sum(mask)
      for m in range(ysize):
          wbftemp = np.where(mask & (np.abs(y-self.ygrid[m]) < ystep/2.0))[0]
          for n in range(xsize):
              wbf = wbftemp[
                  np.where(np.abs(x[wbftemp] - self.xgrid[n]) < xstep/2.0)]
              if len(wbf) < minpt:
                  self.mask[wbf] = False

      # Redefine clipped variables based on minnumpts for IP mapping:
      yfit = y[self.mask]
      xfit = x[self.mask]
      nfit = np.size(xfit)
      self.flux = flux[self.mask]

      if verbose:
          print(f"Light-curve data points:    {ndata:6d}")
          print(f"Light-curve fitting points: {nfit:6d}")

      # Data-point indices corresponding to each knot:
      self.knotpts = np.zeros(nfit, dtype=int)
      # Index (in knotpts array) of first data-point index:
      self.kploc = -np.ones(ysize*xsize, dtype=int)
      # Number of data points per knot:
      self.knotsize = np.zeros(ysize*xsize, dtype=int)

      loc = 0
      # Find knots that have > minpt points:
      for m in range(ysize):
          wbftemp = np.where(np.abs(yfit-self.ygrid[m]) < ystep/2.0)[0]
          pos1 = xfit[wbftemp]
          for n in range(xsize):
              idx = m*xsize + n
              wbf = wbftemp[np.where(np.abs(pos1 - self.xgrid[n]) < xstep/2.0)]
              if len(wbf) >= minpt:
                  # Nearest knot:
                  self.knotsize[idx] = len(wbf)
                  self.kploc[idx] = loc
                  self.knotpts[loc:loc+self.knotsize[idx]] = wbf
                  loc += self.knotsize[idx] # Update loc

      # Index of the knot to the lower-left of a data point:
      self.binloc = (
          np.asarray((yfit-self.ygrid[0])/ystep, int)*xsize +
          np.asarray((xfit-self.xgrid[0])/xstep, int))

      # Compute distance to the four nearest knots:
      self.ydist = np.mod((yfit-self.ygrid[0])/ystep, 1.0) # Dist from bottom
      self.xdist = np.mod((xfit-self.xgrid[0])/xstep, 1.0) # Dist from left

      for m in range(ysize-1):
          wherey = np.where(
              (yfit >= self.ygrid[m]) & (yfit < self.ygrid[m+1]))[0]
          for n in range(xsize-1):
              wherexy = wherey[np.where(
                  (xfit[wherey] >= self.xgrid[n  ]) &
                  (xfit[wherey] <  self.xgrid[n+1]) )]
              # Knot to the lower-left of the data point:
              # If there are no points in one or more bins:
              gridpt = m*xsize + n
              missing_knot = (
                  self.knotsize[gridpt] == 0 or
                  self.knotsize[gridpt+1] == 0 or
                  self.knotsize[gridpt+xsize] == 0 or
                  self.knotsize[gridpt+1+xsize] == 0)
              if len(wherexy) > 0 and missing_knot:
                  # Set distance to nearest bin (nearest-neighbor interp):
                  # Points closer to lower-left knot:
                  if self.knotsize[gridpt] > 0:
                      loc = self.kploc[gridpt]
                      size = self.knotsize[gridpt]
                      pts = np.intersect1d(
                          wherexy, self.knotpts[loc:loc+size], True)
                      self.ydist[pts] = 0
                      self.xdist[pts] = 0
                  # Points closer to lower-right knot:
                  if self.knotsize[gridpt+1] > 0:
                      loc = self.kploc[gridpt+1]
                      size = self.knotsize[gridpt+1]
                      pts = np.intersect1d(
                          wherexy, self.knotpts[loc:loc+size], True)
                      self.ydist[pts] = 0
                      self.xdist[pts] = 1
                  # Points closer to upper-left knot:
                  if self.knotsize[gridpt+xsize] > 0:
                      loc = self.kploc[gridpt+xsize]
                      size = self.knotsize[gridpt+xsize]
                      pts = np.intersect1d(
                          wherexy, self.knotpts[loc:loc+size], True)
                      self.ydist[pts] = 1
                      self.xdist[pts] = 0
                  # Points closer to upper-right knot:
                  if self.knotsize[gridpt+xsize+1] > 0:
                      loc = self.kploc[gridpt+xsize+1]
                      size = self.knotsize[gridpt+xsize+1]
                      pts = np.intersect1d(
                          wherexy, self.knotpts[loc:loc+size], True)
                      self.ydist[pts] = 1
                      self.xdist[pts] = 1
