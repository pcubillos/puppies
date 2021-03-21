# Copyright (c) 2021 Patricio Cubillos
# puppies is open-source software under the MIT license (see LICENSE)

__all__ = [
    "center",
    ]

import os
import time
import shutil
import copy
import multiprocessing as mp

import numpy as np
import astropy.io.fits as fits

from .. import center as pc
from .. import io as io
from .. import plots as pp
from .. import tools as pt


def center(pup, cfile=None):
    """
    Puppies centering driver function.

    This routine takes and parses the input pup and config file,
    and launches a loop for each requested centering method.
    """
    # Current folder:
    here = os.getcwd()
    # Current pup folder:
    cwd = pup.folder

    # Load data:
    data = io.load(pup.datafile, "data") * pup.fluxunits
    uncert = io.load(pup.uncertfile, "uncert") * pup.fluxunits
    mask = io.load(pup.maskfile, "mask")

    # Pre-processing:
    if cfile is not None:
        io.update(pup, cfile)

    # Check all necessary inputs are provided:
    if "centering" not in pup.inputs.keys():
        pt.error("Missing 'centering' user input.")

    if "lag" in pup.centering:
        if pup.aradius == 0 or pup.asize == 0:
            pt.error("Missing 'aradius' or 'asize' least-asymmetry inputs")
        if os.path.isfile(pup.psf) and (pup.psfarad == 0 or pup.psfasize == 0):
            pt.error("Missing 'psfaradius' or 'psfasize' least-asymmetry inputs")

    if "psffit" in pup.centering:
        if pup.psfscale == 0:
            pt.error("Missing 'psfscale' centering user input.")

    nruns = len(pup.centering)
    # Loop over each run:
    for i in range(nruns):
        # Make a copy of the object:
        puppy = copy.copy(pup)
        puppy.centering = pup.centering[i]

        # Move into centering folder:
        puppy.folder = "{:s}/{:s}".format(cwd, puppy.centering)
        os.chdir(cwd)
        if not os.path.exists(puppy.folder):
            os.mkdir(puppy.folder)
        os.chdir(puppy.folder)

        # Launch the thread:
        centering(puppy, data, uncert, mask)

    # Return to original location:
    os.chdir(here)
    #return list_of_puppies_for_next_step


def centering(pup, data, uncert, mask):
    """
    Compute centering on the PSF and median images, and launch
    multiprocessing thread to compute the centering on the individual
    frames.  Finally, add centering variables into the pup object.
    """
    # Copy, update, and reopen logfile:
    shutil.copy(pup.logfile, pup.folder)
    pup.logfile = f"{pup.folder}/{pup.ID}.log"
    pup.log = open(pup.logfile, "a")
    pt.msg(1,
        f"\n\n{70*':'}\n"
        f"Starting {pup.centering} centering ({time.ctime()})\n\n", pup.log)

    # Check least asym parameters work:
    if pup.centering in ['lac', 'lag']:
        if pup.ctrim < (pup.aradius + pup.asize) and pup.ctrim != 0:
            pup.ctrim = pup.aradius + pup.asize + 3
            pt.msg(1,
                f'Trim radius is too small, changed to: {pup.ctrim}', pup.log)
        if pup.psftrim < (pup.psfarad + pup.psfasize) and pup.psftrim !=0:
            pup.psftrim = pup.psfarad + pup.psfasize + 3
            pt.msg(1,
                f'PSF Trim radius is too small, changed to: {pup.psftrim}',
                pup.log)

    # PSF Centering:
    if os.path.isfile(pup.psf):
        pup.psfim, psfheader = fits.getdata(
            pup.psf, header=True, ignore_missing_end=True)
        # Guess of the center of the PSF (center of psfim)
        psfctrguess = np.asarray(np.shape(pup.psfim))/2
        # Find center of PSF:
        pup.psfctr, extra = pc.center(
            pup.centering, pup.psfim, psfctrguess,
            pup.psftrim, pup.psfarad, pup.psfasize)
        pt.msg(1, f'PSF center found at: {pup.psfctr}', pup.log)
    else:
        pup.psfim = None
        pup.psfctr = None
        pt.msg(1, 'No PSF supplied.', pup.log)

    # Find center of the mean Image:
    pup.targpos = np.zeros((2, pup.inst.npos))
    for pos in range(pup.inst.npos):
        meanim = pup.meanim[pos]
        guess = pup.srcest[:, pos]
        targpos, extra = pc.center(
            pup.centering, meanim, guess, pup.ctrim,
            pup.aradius, pup.asize, fitbg=pup.fitbg,
            psf=pup.psfim, psfctr=pup.psfctr, expand=pup.psfscale)
        pup.targpos[:,pos] = targpos
    pt.msg(1,
        f"Center position(s) of the mean Image(s):\n{pup.targpos.T}", pup.log)

    # Multy Process set up:
    x     = mp.Array("d", np.zeros(pup.inst.nframes))
    y     = mp.Array("d", np.zeros(pup.inst.nframes))
    flux  = mp.Array("d", np.zeros(pup.inst.nframes))
    sky   = mp.Array("d", np.zeros(pup.inst.nframes))
    good  = mp.Array("b", np.zeros(pup.inst.nframes, bool))

    # Size of chunk of data each core will process:
    chunksize = int(pup.inst.nframes/pup.ncpu + 1)
    pt.msg(1, f"Number of parallel CPUs: {pup.ncpu}.", pup.log)

    # Start Muti Procecess:
    processes = []
    for n in range(pup.ncpu):
        start =  n    * chunksize # Starting index to process
        end   = (n+1) * chunksize # Ending   index to process
        args = (pup, data, uncert, mask, start, end, x, y, flux, sky, good)
        proc = mp.Process(target=calc_center, args=args)
        processes.append(proc)
        proc.start()

    # Make sure all processes finish their work:
    for n in range(pup.ncpu):
        processes[n].join()

    # Put the results in the object. I need to reshape them:
    pup.fp.x = np.array(x)
    pup.fp.y = np.array(y)
    pup.fp.goodcen = np.array(good, bool)
    # Flag out out of boundaries y,x frames:
    pup.fp.goodcen[
        (pup.fp.y<0) | (pup.fp.y>pup.inst.ny) |
        (pup.fp.x<0) | (pup.fp.x>pup.inst.nx) ] = False
    # goodcen is the centering good flag, good is current to each step:
    pup.fp.good = np.copy(pup.fp.goodcen)

    # If PSF fit:
    if pup.centering in ["ipf", "bpf"]:
        # FINDME: This line will break
        pup.fp.aplev  = np.array(aplev)
        pup.fp.skylev = np.array(sky)

    pup.fp.r = np.zeros(pup.inst.nframes)
    for pos in range(pup.inst.npos):
        igood = (pup.fp.pos==pos) & pup.fp.good
        ymean = np.round(np.median(pup.fp.y[igood]))
        xmean = np.round(np.median(pup.fp.x[igood]))
        # Distance to center of mean pixel:
        ipos = pup.fp.pos==pos
        pup.fp.r[ipos] = np.sqrt(
            (pup.fp.y[ipos]-ymean)**2.0 + (pup.fp.x[ipos]-xmean)**2.0)

        # YX stats:
        dy = np.ediff1d(pup.fp.y[igood])
        dx = np.ediff1d(pup.fp.x[igood])
        pup.yrms = np.sqrt(np.mean(dy**2))
        pup.xrms = np.sqrt(np.mean(dx**2))

        mean_y, mean_x = np.mean(pup.fp.y[igood]), np.mean(pup.fp.x[igood])
        std_y, std_x = np.std( pup.fp.y[igood]), np.std( pup.fp.x[igood])
        mean_dy, mean_dx = np.mean(np.abs(dy)), np.mean(np.abs(dx))
        med_dy, med_dx = np.median(np.abs(dy)), np.median(np.abs(dx))
        pt.msg(1,
            f"\nPosition {pos:2d}:   Y (pixel)   X (pixel)\n"
            f"mean          {mean_y:10.5f}  {mean_x:10.5f}\n"
            f"std dev.      {std_y:10.5f}  {std_x:10.5f}\n"
            f"RMS(delta)    {pup.yrms:10.5f}  {pup.xrms:10.5f}\n"
            f"mean(delta)   {mean_dy:10.5f}  {mean_dx:10.5f}\n"
            f"median(delta) {med_dy:10.5f}  {med_dx:10.5f}", pup.log)

    # Plots:
    pp.yx(pup.fp.y, pup.fp.x, pup.fp.phase, pup.fp.good, pup.fp.pos, pup.folder)

    # Print time stamp, save, and close:
    pt.msg(1,
        f"\nFinished {pup.centering} centering ({time.ctime()})."
        f"\nOutput folder: '{pup.folder}'.\n", pup.log)
    io.save(pup)


def calc_center(pup, data, uncert, mask, start, end, x, y, flux, sky, good):
    """
    Multiprocessing child routine to compute centering on a set of
    frames.
    """
    # Initialize a Timer to report progress:
    if start == 0:  # Only for the fisrt chunk
        #clock = t.Timer(pup.npos*end,
        #                progress=np.array([0.05, 0.1, 0.2, 0.3, 0.4,  0.5,
        #                                   0.6,  0.7, 0.8, 0.9, 0.99, 1.1]))
        pass

    unc = None
    # Recalculate end, care not to go out of bounds:
    end = np.amin([end, pup.inst.nframes])
    # Compute the centering in each frame:
    for i in range(start, end):
        pos = pup.fp.pos[i]
        try:
            if pup.cweights:   # weight by uncertainties in fitting?
                unc = uncert[i]
            # Do the centering:
            position, extra = pc.center(
                pup.centering, data[i], pup.targpos[:,pos], pup.ctrim,
                pup.aradius, pup.asize, mask[i], unc, fitbg=pup.fitbg,
                expand=pup.psfscale, psf=pup.psfim, psfctr=pup.psfctr)
            y[i], x[i] = position
            good[i] = True
            # Not necessarily true, it just means centering didn't crashed
            if pup.centering in ["ipf", "bpf"]:
                flux[i] = extra[0]
                sky [i] = extra[1]
                # FINDME: define some criterion for good/bad fit.
        except:
            y[i], x[i] = pup.targpos[:, pos]
            flux[i], sky[i] = 0.0, 0.0
            good[i] = False
            pt.msg(1, "Centering failed in frame {:d}.".format(i), pup.log)

        if start == 0:
            #print("{}/{}".format(i,end))
            # Report progress:
            #clock.check(pos*end + i, name=pup.folder)
            pass
