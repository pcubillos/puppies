import sys
import os
import matplotlib.pyplot as plt
import numpy as np

from .. import stats as ps

rootdir = os.path.realpath(os.path.dirname(__file__) + "/../../")
sys.path.append(rootdir + "/modules/MCcubed/")
import MCcubed.utils as mu

__all__ =["yx", "background", "rawflux"]


def yx(y, x, phase=None, good=None, pos=None, folder=None):
  """
  Plot y,x positions as a function of phase.

  Parameters
  ----------
  y: 1D float ndarray
     Frame y pixel positions.
  x: 1D float ndarray
     Frame x pixel positions.
  phase: 1D float ndarray
     Frame orbital phase.
  good: 1D bool ndarray
     Good frame flag (True=good, False=bad).
  pos: 1D integer ndarray
     Frame pointing position of the telescope.
  folder: String
     Output folder where to save the plot.
  """
  if pos is None:
    pos = np.zeros(len(y), int)

  if phase is None:
    phase = np.arange(len(y), int)

  npos = len(np.unique(pos))
  suffix = ""

  for j in np.unique(pos):
    # Append suffix to filename in case there is more than one position:
    if npos > 1:
      suffix = "_pos{:02d}".format(pos)

    ipos  = pos == j
    igood = ipos &  good
    ibad  = ipos & ~good
    yy = y[ipos]
    xx = x[ipos]

    # X-axis' range:
    dt = np.ptp(phase[igood])
    tran = np.amin(phase[igood]) - 0.025*dt, np.amax(phase[igood]) + 0.025*dt

    plt.figure(503)
    plt.clf()
    # Figure adjusted to include all good frames:
    ax = plt.subplot(211)
    plt.plot(phase[igood], y[igood], '.', color="b",      zorder=1, ms=2)
    yran = ax.get_ylim()
    plt.plot(phase[ibad],  y[ibad],  '.', color="orange", zorder=0, ms=2)
    plt.ylabel("Y (pixels)")
    plt.ylim(yran)
    plt.xlim(tran)

    ax = plt.subplot(212)
    plt.plot(phase[igood], x[igood], '.', color="b",      zorder=1, ms=2)
    yran = ax.get_ylim()
    plt.plot(phase[ibad],  x[ibad],  '.', color="orange", zorder=0, ms=2)
    plt.ylabel("X (pixels)")
    plt.xlabel('Orbital phase')
    plt.ylim(yran)
    plt.xlim(tran)
    if folder is not None:
      plt.savefig(folder + "/yx{:s}.png".format(suffix))

    # Excluding outliers (even though they might not have been flagged):
    s = 6  # Number of standard deviations wide
    iy = (y>np.percentile(y[igood], 5)) & (y<np.percentile(y[igood], 95))
    ix = (x>np.percentile(x[igood], 5)) & (x<np.percentile(x[igood], 95))
    yran = np.mean(y[iy & igood]) - s*np.std(y[iy & igood]), \
           np.mean(y[iy & igood]) + s*np.std(y[iy & igood])
    xran = np.mean(x[ix & igood]) - s*np.std(x[ix & igood]), \
           np.mean(x[ix & igood]) + s*np.std(x[ix & igood])
    ax = plt.subplot(211)
    plt.ylim(yran)
    ax = plt.subplot(212)
    plt.ylim(xran)
    if folder is not None:
      plt.savefig(folder + "/yx_zoom{:s}.png".format(suffix))


def background(skylev, phase=None, good=None, folder=None, units='units'):
  """
  Make sky background plot.

  Parameters
  ----------
  skylev: 1D float ndarray
     Frames sky level.
  phase: 1D float ndarray
     Frames orbital phase.
  good: 1D bool ndarray
     Good frame mask.
  folder: String
     Folder where to store the output plots.
  units: String
     Flux units.
  """
  if phase is None:
    phase = np.arange(len(y), int)

  # X-axis' range:
  dt = np.ptp(phase[good])
  xran = np.amin(phase[good]) - 0.025*dt, np.amax(phase[good]) + 0.025*dt

  ms = 3.0
  plt.figure(505)
  plt.clf()
  ax = plt.subplot(111)
  plt.plot(phase[ good], skylev[ good], ".", color="b",      zorder=1, ms=ms)
  yran = ax.get_ylim()
  plt.plot(phase[~good], skylev[~good], ".", color="orange", zorder=0, ms=ms)
  plt.ylim(yran)
  plt.xlim(xran)
  plt.ylabel("Sky background ({:s}/pixel)".format(units))
  plt.xlabel('Orbital phase')

  suffix = ""
  if folder is not None:
    plt.savefig(folder + "/sky_background{:s}.png".format(suffix))


def rawflux(flux, ferr, phase=None, good=None, folder=None,
            sigrej=None, binsize=None, units='units'):
  """
  Make raw flux plots (all individual frames, all frames without
  outliers, and binned frames).

  Parameters
  ----------
  flux: 1D float ndarray
     Frames flux values.
  ferror: 1D float ndarray
     Frames flux uncertainties.
  phase: 1D float ndarray
     Frames orbital phase.
  good: 1D bool ndarray
     Good frame mask.
  folder: String
     Folder where to store the output plots.
  sigrej: 1D float array
     Sigma rejection threshold.
  binsize: Integer
     Binning bin size (defaulted to 400 points per orbit).
  units: String
     Flux units.
  """
  if phase is None:
    phase = np.arange(len(y), int)

  # X-axis' range:
  dt = np.ptp(phase[good])
  xran = np.amin(phase[good]) - 0.025*dt, np.amax(phase[good]) + 0.025*dt

  ms = 3.0
  plt.figure(504)
  plt.clf()
  ax = plt.subplot(111)
  plt.plot(phase[ good], flux[ good], ".", color="b",      zorder=1, ms=ms)
  yran = ax.get_ylim()
  plt.plot(phase[~good], flux[~good], ".", color="orange", zorder=0, ms=ms)
  plt.ylim(yran)
  plt.xlim(xran)
  plt.ylabel("Raw flux ({:s})".format(units))
  plt.xlabel('Orbital phase')

  if folder is not None:
    plt.savefig(folder + "/raw_flux.png")

  # Mask out outliers and get yran:
  if sigrej is None:
    sigrej = [5,5,5]
  mask = ps.sigrej(flux, sigrej, mask=np.copy(good))
  plt.clf()
  ax = plt.subplot(111)
  plt.plot(flux[mask])
  yran = ax.get_ylim()

  # Replot with the narrower yran:
  plt.clf()
  plt.plot(phase[ good], flux[ good], ".", color="b",      zorder=1, ms=ms)
  plt.plot(phase[~good], flux[~good], ".", color="orange", zorder=0, ms=ms)
  plt.ylim(yran)
  plt.xlim(xran)
  plt.ylabel("Raw flux without outliers ({:s})".format(units))
  plt.xlabel('Orbital phase')
  if folder is not None:
    plt.savefig(folder + "/raw_flux_zoom.png")

  # Binned flux:
  if binsize is None:
    # Draw 400 points per orbit:
    binsize = int(np.sum(good)/dt / 400.0)
  binflux, binunc, binphase = mu.binarray(flux[good], ferr[good],
                                          phase[good], binsize)

  plt.clf()
  ax = plt.subplot(111)
  plt.errorbar(binphase, binflux, binunc, fmt="bo")
  plt.xlim(xran)
  plt.ylabel("Binned raw flux ({:s})".format(units))
  plt.xlabel('Orbital phase')
  if folder is not None:
    plt.savefig(folder + "/raw_flux_binned.png")


def yxflux():
  """
aplev = pup.fp.aplev
y = pup.fp.y
x = pup.fp.x
good = pup.fp.good
pos = pup.fp.pos
  """
  if pos is None:
    pos = np.zeros(len(y), int)

  if phase is None:
    phase = np.arange(len(y), int)

  for j in np.unique(pos):
    # Append suffix to filename in case there is more than one position:
    if npos > 1:
      suffix = "_pos{:02d}".format(pos)

    ipos  = pos == j
    igood = ipos &  good
    ibad  = ipos & ~good
    yy = y[ipos]
    xx = x[ipos]

    plt.figure(502)
    plt.clf()
    plt.subplot(211)
    plt.plot(y[igood], aplev[igood], "b.") #, binyapstd, fmt=fmt1[pos])
    plt.ylabel('Flux')
    plt.subplot(212)
    plt.plot(x[igood], aplev[igood], "r.") #, binxapstd, fmt=fmt1[pos],
    plt.xlabel('Pixel Postion')
    plt.ylabel('Flux')


  plt.figure(504)
  plt.errorbar(binrr, binraplev, binrapstd, fmt=fmt1[pos],
               label=('pos %i'%(pos)))
  plt.title(pup.planetname + ' Radial Distance vs. Flux')
  plt.xlabel('Distance From Center of Pixel')
  plt.ylabel('Flux')
  plt.legend(loc='best')


def yxdensity(y, x, dy, dx, minpt=1):
  """
y = pup.fp.y
x = pup.fp.x
dy = pup.yrms
dx = pup.xrms
minpt=1

  knotpts:   ndata
    Data-point indices sorted by knot.
  knotsize:  nknots
    Number of datapoints per knot.
  kploc:     nknots
    Index of first data-point index of each knot.
  binloc:    ndata
    Index of the knot to the lower left of the data points.
  """
  blissmask, ygrid, xgrid, knotpts, knotsize, kploc, binloc, \
    ydist, xdist = bliss.setup(y, x, dy, dx, minpt, True)
  knotdens = knotsize.reshape((len(ygrid), len(xgrid)))

  palette = plt.cm.plasma
  palette.set_under(alpha=0.0, color='w')

  dy = ygrid[1] - ygrid[0]
  dx = xgrid[1] - xgrid[0]
  bottom, top = ygrid[0] - 0.5*dy, ygrid[-1] + 0.5*dy
  left, right = xgrid[0] - 0.5*dx, xgrid[-1] + 0.5*dx
  # FINDME: need to add the reference (integer) pixel.

  plt.figure(509)
  plt.clf()
  plt.imshow(knotdens, interpolation="nearest", origin="lower",
             extent=(left, right, bottom, top), cmap=palette, vmin=1)
  plt.xlabel("X (pixels)")
  plt.ylabel("Y (pixels)")
  cb = plt.colorbar()
  cb.set_label("Number of points")

  n = len(ygrid)*len(xgrid)
  knotflux = np.zeros(n)
  for i in np.arange(n):
    if knotsize[i] > 0:
      indices = knotpts[kploc[i]:kploc[i]+knotsize[i]]
      knotflux[i] = np.median(aplev[indices])
  knotflux = knotflux.reshape((len(ygrid), len(xgrid)))
  plt.figure(510)
  plt.clf()
  plt.imshow(knotflux, interpolation="nearest", origin="lower",
             extent=(left, right, bottom, top), cmap=palette,
             vmin=np.amin(knotflux[knotflux>0]))
  plt.xlabel("X (pixels)")
  plt.ylabel("Y (pixels)")
  cb = plt.colorbar()
  cb.set_label("Median flux")

