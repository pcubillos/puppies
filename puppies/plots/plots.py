import matplotlib.pyplot as plt
import numpy as np

__all__ =["yx"]


def yx(y, x, time=None, good=None, pos=None, folder=None):
  """
  Plot y,x positions as a function of time/phase.

  Parameters
  ----------
  y: 1D float ndarray
     Frame y pixel positions.
  x: 1D float ndarray
     Frame x pixel positions.
  time: 1D float ndarray
     Frame time stamp (phase preferred).
  good: 1D bool ndarray
     Good frame flag (True=good, False=bad).
  pos: 1D integer ndarray
     Frame pointing position of the telescope.
  folder: String
     Output folder where to save the plot.
  """
  if pos is None:
    pos = np.zeros(len(y), int)

  if time is None:
    time = np.arange(len(y), int)

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
    dt = np.ptp(time[igood])
    tran = np.amin(time[igood]) - 0.025*dt, np.amax(time[igood]) + 0.025*dt

    plt.figure(503)
    plt.clf()
    # Figure adjusted to include all good frames:
    ax = plt.subplot(211)
    plt.plot(time[igood], y[igood], '.', color="b",      zorder=1, ms=2)
    yran = ax.get_ylim()
    plt.plot(time[ibad],  y[ibad],  '.', color="orange", zorder=0, ms=2)
    plt.ylabel("Y (pixels)")
    plt.ylim(yran)
    plt.xlim(tran)

    ax = plt.subplot(212)
    plt.plot(time[igood], x[igood], '.', color="b",      zorder=1, ms=2)
    yran = ax.get_ylim()
    plt.plot(time[ibad],  x[ibad],  '.', color="orange", zorder=0, ms=2)
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
