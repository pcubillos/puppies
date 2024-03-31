# Copyright (c) 2021-2024 Patricio Cubillos
# puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

__all__ = [
    'yx',
    'background',
    'rawflux',
    'lightcurve',
]

import os

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import mc3

from .. import stats as ps
from ..models import bliss


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

        # X-axis' range:
        dt = np.ptp(phase[igood])
        tran = (
            np.amin(phase[igood]) - 0.025*dt,
            np.amax(phase[igood]) + 0.025*dt,
        )

        plt.figure(503)
        plt.clf()
        # Figure adjusted to include all good frames:
        ax = plt.subplot(211)
        plt.plot(phase[igood], y[igood], '.', color="blue", zorder=1, ms=2)
        yran = ax.get_ylim()
        plt.plot(phase[ibad], y[ibad], '.', color="orange", zorder=0, ms=2)
        plt.ylabel("Y (pixels)")
        plt.ylim(yran)
        plt.xlim(tran)

        ax = plt.subplot(212)
        plt.plot(phase[igood], x[igood], '.', color="blue", zorder=1, ms=2)
        yran = ax.get_ylim()
        plt.plot(phase[ibad], x[ibad], '.', color="orange", zorder=0, ms=2)
        plt.ylabel("X (pixels)")
        plt.xlabel('Orbital phase')
        plt.ylim(yran)
        plt.xlim(tran)
        if folder is not None:
            plt.savefig(f"{folder}/yx{suffix}.png")

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
            plt.savefig(f"{folder}/yx_zoom{suffix}.png")


def background(skylev, phase, good=None, folder=None, units='units'):
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
        plt.savefig(f"{folder}/sky_background{suffix}.png")


def rawflux(
        flux, ferr, phase, good=None, folder=None,
        sigrej=None, binsize=None, units='units',
    ):
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
        plt.savefig(f"{folder}/raw_flux.png")

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
    plt.ylabel(f"Raw flux without outliers ({units})")
    plt.xlabel('Orbital phase')
    if folder is not None:
        plt.savefig(f"{folder}/raw_flux_zoom.png")

    # Binned flux:
    if binsize is None:
        # Draw 400 points per orbit:
        binsize = int(np.sum(good)/dt / 400.0)
    binflux, binunc = mc3.stats.bin_array(
        flux[good], binsize, ferr[good])
    binphase = mc3.stats.bin_array(phase[good], binsize)

    plt.clf()
    ax = plt.subplot(111)
    plt.errorbar(binphase, binflux, binunc, fmt="bo")
    plt.xlim(xran)
    plt.ylabel(f"Binned raw flux ({units})")
    plt.xlabel('Orbital phase')
    if folder is not None:
        plt.savefig(f"{folder}/raw_flux_binned.png")


def yxflux(x, y, flux, phase, good, position=None, folder=None):
    """
    Plot flux vs X position, and flux vs Y position
    """
    if position is None:
        position = np.zeros(len(y), int)

    if phase is None:
        phase = np.arange(len(y), int)

    npos = len(np.unique(position))
    for j in np.unique(position):
        # Append suffix to filename in case there is more than one position:
        suffix = f"_pos{position:02d}" if npos > 1 else ''

        igood = (position == j) & good
        #ibad = (position == j) & ~good

        plt.figure(502)
        plt.clf()
        plt.subplot(211)
        plt.plot(y[igood], flux[igood], "b.")
        plt.ylabel('Flux')
        plt.subplot(212)
        plt.plot(x[igood], flux[igood], "r.")
        plt.xlabel('Pixel Postion')
        plt.ylabel('Flux')
        if folder is not None:
            plt.savefig(f"{folder}/flux_vs_xy{suffix}.png")


def yxdensity(y, x, dy, dx, aplev, minpt=1):
    """
    y = pup.fp.y
    x = pup.fp.x
    dy = pup.yrms
    dx = pup.xrms
    minpt=1

    knotpts: ndata
        Data-point indices sorted by knot.
    knotsize: nknots
        Number of datapoints per knot.
    kploc: nknots
        Index of first data-point index of each knot.
    binloc: ndata
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
    plt.imshow(
        knotdens, interpolation="nearest", origin="lower",
        extent=(left, right, bottom, top), cmap=palette, vmin=1,
    )
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    cb = plt.colorbar()
    cb.set_label("Number of points")

    n = len(ygrid)*len(xgrid)
    knotflux = np.zeros(n)
    for i in range(n):
        if knotsize[i] > 0:
            indices = knotpts[kploc[i]:kploc[i]+knotsize[i]]
            knotflux[i] = np.median(aplev[indices])
    knotflux = knotflux.reshape((len(ygrid), len(xgrid)))
    plt.figure(510)
    plt.clf()
    plt.imshow(
        knotflux,
        interpolation="nearest", origin="lower",
        extent=(left, right, bottom, top), cmap=palette,
        vmin=np.amin(knotflux[knotflux>0]),
    )
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    cb = plt.colorbar()
    cb.set_label("Median flux")


def lightcurve(fit, pups, savefile=None, systematics='raw'):
    for j,ipup in enumerate(fit.ipup):
        pup = pups[ipup]
        models = fit.mnames[j]

        iflux = fit.models[j][0].pnames.index('flux')
        sflux = fit.models[j][0].params[iflux]
        astro_pars = np.copy(fit.models[j][0].params)
        astro_pars[iflux] = 1.0
        model_fit = fit.models[j][0](astro_pars)

        ipflux = 1.0
        if 'bliss' in models:
            idx_bliss = models.index('bliss')
            ipflux = fit.models[j][idx_bliss].ipflux

        ramp = 1.0
        for model in models:
            if model.endswith('ramp'):
                idx_ramp = models.index(model)
                ramp = fit.models[j][idx_ramp](fit.models[j][idx_ramp].params)
                break

        data = pup.flux[fit.mask[j]]/sflux
        uncert = pup.ferr[fit.mask[j]]/sflux
        time = pup.time[fit.mask[j]]
        bestfit = fit.bestfit[j]/sflux

        if systematics == 'raw':
            model_fit *= ipflux * ramp
        elif systematics == 'ramp':
            model_fit *= ramp
            data /= ipflux
            uncert /= ipflux
            bestfit /= ipflux
        elif systematics == 'corrected':
            data /= ipflux*ramp
            uncert /= ipflux*ramp
            bestfit /= ipflux*ramp

        binsize = int(fit.ndata[j]/pup.nbins)
        binflux, binferr = mc3.stats.bin_array(data, binsize, uncert)
        bintime = mc3.stats.bin_array(time, binsize)

        col_bin = 'blue'
        col_fit = 'orangered'
        col_pt = to_rgba('0.65', alpha=0.45)
        plt.figure(303, (7,6))
        plt.clf()
        plt.subplots_adjust(0.12, 0.12, 0.95, 0.95)
        ax = plt.subplot(211)
        ax.plot(time, data, ".", ms=3, color=col_pt)
        ax.errorbar(bintime, binflux, binferr, fmt=".", color=col_bin, zorder=3)
        #ax.plot(time, bestfit, color=col_fit, zorder=4)
        ax.plot(time, model_fit, color=col_fit, zorder=4)
        ax.set_ylabel("Normalized flux")
        ax.tick_params(which='both', right=True, top=True, direction='in')
        ax.set_xlim(np.amin(time), np.amax(time))

        ax = plt.subplot(212)
        ax.errorbar(bintime, binflux, binferr, fmt=".", color=col_bin, zorder=3)
        #ax.plot(time, bestfit, color=col_fit, zorder=4)
        ax.plot(time, model_fit, color=col_fit, zorder=4)
        ylims = ax.get_ylim()
        ax.plot(time, data, ".", ms=3, color=col_pt)
        ax.tick_params(which='both', right=True, direction='in')
        ax.set_ylim(ylims)
        ax.set_xlim(np.amin(time), np.amax(time))
        ax.set_xlabel("Orbital phase")
        ax.set_ylabel("Normalized flux")
        if savefile is not None:
            if fit.npups > 1:
                path, ext = os.path.splitext(savefile)
                #savefile = savefile_lc.replace('_lc_', f'_pup{j}_lc_')
                plt.savefig(f'{path}_pup{j:02}{ext}', dpi=300)
            else:
                plt.savefig(savefile, dpi=300)

