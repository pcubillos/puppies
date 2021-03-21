# Copyright (c) 2021 Patricio Cubillos
# puppies is open-source software under the MIT license (see LICENSE)

__all__ = [
    "photom",
     ]

import os
import time
import shutil
import copy
import multiprocessing as mp

import numpy as np
#import timer       as t
#import psf_fit     as pf
#import optphot     as op

from .. import image as im
from .. import io as io
from .. import photometry as ph
from .. import plots as pp
from .. import tools as pt


def update(pup, cfile):
  """
  Update user parameters for photometry from configuration file,
  and check that all necessary values are well defined.
  """

def photom(pup, cfile=None):
    """
    Load the event.
    Read config file.
    Launch a thread for each centering run.
    """
    # Current folder:
    here = os.getcwd()
    # Current pup folder:
    cwd = pup.folder

    # Load the event data:
    data = io.load(pup.datafile, "data")
    uncert = io.load(pup.uncertfile, "uncert")
    mask = io.load(pup.maskfile, "mask")

    # Pre-processing:
    if cfile is not None:
        io.update(pup, cfile)

    # Required inputs:
    if not hasattr(pup, 'photap'):
        pt.error("Missing 'photap' user input.")
    if not hasattr(pup, 'skyin'):
        pt.error("Missing 'skyin' user input.")
    if not hasattr(pup, 'skyout'):
        pt.error("Missing 'skyout' user input.")

    # Check aperture photometry is a positive float or 'optimal':
    for i in range(len(pup.photap)):
        try:
            pup.photap[i] = float(pup.photap[i])
            if pup.photap[i] <= 0.0:
                pt.error(f"Invalid photap aperture: {pup.photap[i]:.2f}.")
        except ValueError:
            if pup.photap[i] != "optimal":
                pt.error(f"Invalid photap entry: '{pup.photap[i]}'.")
    # Check sky annuli inputs have same length than photap:
    if len(pup.skyin) != len(pup.photap):
        if len(pup.skyin) != 1:
            pt.error("'skyin' and 'photap' have inconsistent lengths.")
        pup.skyin = np.repeat(pup.skyin, len(pup.photap))
    if len(pup.skyout) != len(pup.photap):
        if len(pup.skyout) != 1:
            pt.error("'skyout' and 'photap' have inconsistent lengths.")
        pup.skyout = np.repeat(pup.skyout, len(pup.photap))


    nruns = len(pup.photap)
    # Loop over each run:
    for i in range(nruns):
        # Make a copy of the event:
        puppy = copy.copy(pup)
        puppy.photap = pup.photap[i]
        puppy.skyin  = pup.skyin [i]
        puppy.skyout = pup.skyout[i]

        # Folder suffix when necessary:
        folder = ""
        if puppy.photap != 'optimal':
            folder += (
                f'aper{puppy.photap*100:03.0f}'
                f'{puppy.skyin:02.0f}{puppy.skyout:02.0f}')
        else:
            folder = puppy.photap

        # Move into photometry folder:
        puppy.folder = f"{puppy.folder}/{folder}"
        os.chdir(cwd)
        if not os.path.exists(puppy.folder):  # Create directory if necessary
            os.mkdir(puppy.folder)
        os.chdir(puppy.folder)

        # Launch the thread:
        photometry(puppy, data, uncert, mask)

    # Return to original location:
    os.chdir(here)
    #return list_of_puppies_for_next_step


def photometry(pup, data, uncert, mask):
    """
    Doc me.
    """
    #tini = time.time()

    if isinstance(pup.photap, float):
        photom = f"aperture {pup.photap:.2f}"
    else:
        photom = pup.photap

    # Copy, update, and reopen logfile:
    shutil.copy(pup.logfile, pup.folder)
    pup.logfile = "{:s}/{:s}.log".format(pup.folder, pup.ID)
    pup.log = open(pup.logfile, "a")
    pt.msg(1,
        f"\n\n{70*':'}\nStarting {photom} photometry  ({time.ctime()})\n\n",
        pup.log)

    # Copy photom.pcf in photdir
    #pcf.make_file("photom.pcf")

    nframes = pup.inst.nframes
    # Aperture photometry:
    if pup.photap != "optimal":
        # Multiprocess set up:
        aplev = mp.Array("d", np.zeros(nframes))  # aperture flux
        aperr = mp.Array("d", np.zeros(nframes))  # aperture error
        nappix = mp.Array("d", np.zeros(nframes))  # number of aperture pixels
        skylev = mp.Array("d", np.zeros(nframes))  # sky level
        skyerr = mp.Array("d", np.zeros(nframes))  # sky error
        nskypix = mp.Array("d", np.zeros(nframes))  # number of sky pixels
        nskyideal = mp.Array("d", np.zeros(nframes))  # ideal sky pixels
        status = mp.Array("i", np.zeros(nframes, int))   # apphot return status
        good   = mp.Array("b", np.zeros(nframes, bool))  # good photometry flag
        # FINDME: Move this allocation out of the if?

        # Size of chunk of data each core will process:
        chunksize = int(nframes/pup.ncpu + 1)
        pt.msg(1,
            f"Number of parallel CPUs for photometry: {pup.ncpu}.", pup.log)

        # Start Muti Procecess:
        processes = []
        for n in range(pup.ncpu):
            start =  n    * chunksize
            end   = (n+1) * chunksize
            args = (
                pup, data, uncert, mask, start, end, aplev, aperr, nappix,
                skylev, skyerr, nskypix, nskyideal, status, good)
            proc = mp.Process(target=calc_photom, args=args)
            processes.append(proc)
            proc.start()
        # Make sure all processes finish their work:
        for n in range(pup.ncpu):
            processes[n].join()

        # Put the results in the event. I need to reshape them:
        pup.fp.aplev     = np.array(aplev)
        pup.fp.aperr     = np.array(aperr)
        pup.fp.nappix    = np.array(nappix)
        pup.fp.skylev    = np.array(skylev)
        pup.fp.skyerr    = np.array(skyerr)
        pup.fp.nskypix   = np.array(nskypix)
        pup.fp.nskyideal = np.array(nskyideal)
        pup.fp.status    = np.array(status)
        pup.fp.goodphot  = np.array(good, bool)
        # Overwrite good with goodphot:
        pup.fp.good = np.copy(pup.fp.goodphot)

        # Raw photometry (star + sky flux within the aperture):
        pup.fp.apraw = pup.fp.aplev + pup.fp.skylev*pup.fp.nappix

    # FINDME: Make this a multiprocessing task as well.
    elif pup.photap == "optimal":
        # utils for profile construction:
        pshape = np.array([2*pup.otrim+1, 2*pup.otrim+1])
        subpsf = np.zeros(np.asarray(pshape, float)*pup.expand)
        #x = np.indices(pshape)
        #clock = t.Timer(np.sum(pup.nimpos),
        #                progress=np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1.1]))

        for i in range(nframes):
            # Integer part of center of subimage:
            cen = np.rint([pup.fp.y[i], pup.fp.x[i]])
            # Center in the trimed image:
            loc = (pup.otrim, pup.otrim)
            # Do the trim:
            img, msk, err = im.trim(
                pup.data[i], cen, loc, mask=pup.mask[i], uncert=uncert[i])

            # Center of star in the subimage:
            ctr = (
                pup.fp.y[i]-cen[0]+pup.otrim,
                pup.fp.x[i]-cen[1]+pup.otrim)
            # Make profile:
            # Index of the position in the supersampled PSF:
            pix = pf.pos2index(ctr, pup.expand)
            profile, pctr = pf.make_psf_binning(
                pup.psfim, pshape, pup.expand,
                [pix[0], pix[1], 1.0, 0.0],
                pup.psfctr, subpsf)

            # Subtract the sky level:
            img -= pup.fp.skylev[i]
            # optimal photometry calculation:
            immean, imerr, good = op.optphot(
                img, profile, var=err**2.0, mask=msk)
            # FINDME: Not fitting the sky at the same time? I dont like this

            pup.fp.aplev[i] = immean
            pup.fp.aperr[i] = imerr
            pup.fp.skylev[i] = pup.fp.skylev[i]
            pup.fp.good[i] = good

            # Report progress:
            #clock.check(np.sum(pup.nimpos[0:pos]) + i, name=pup.folder)

    # Print results into the log:
    for i in range(nframes):
        pt.msg(1, '\nframe = {:11d}  pos ={:3d}  status ={:3d}  good ={:3d}\n'
          'aplev = {:11.3f}  skylev = {:7.3f}  nappix    = {:7.2f}\n'
          'aperr = {:11.3f}  skyerr = {:7.3f}  nskypix   = {:7.2f}\n'
          'y     = {:11.3f}  x      = {:7.3f}  nskyideal = {:7.2f}\n'
           .format(i, pup.fp.pos[i], pup.fp.status[i], pup.fp.good[i],
             pup.fp.aplev[i], pup.fp.skylev[i], pup.fp.nappix [i],
             pup.fp.aperr[i], pup.fp.skyerr[i], pup.fp.nskypix[i],
             pup.fp.y[i],     pup.fp.x[i],      pup.fp.nskyideal[i]),
               pup.log)

    if pup.centering in ["bpf"]:
        pup.ispsf = False

    if os.path.isfile(pup.psf) and isinstance(pup.photap, float):
        # PSF aperture correction:
        pt.msg(1, 'Calculating PSF aperture.', pup.log)
        pup.psfim = pup.psfim.astype(np.float64)

        imerr = np.ones(np.shape(pup.psfim))
        imask = np.ones(np.shape(pup.psfim), bool)
        skyfrac = 0.1
        pup.psffrac, aperr, psfnappix,  pup.psfskylev, skyerr, \
           psfnskypix, psfnskyideal, pup.psfstatus  \
                       = ph.aphot(pup.psfim, imerr, imask,
                                     pup.psfctr[0], pup.psfctr[1],
                                     pup.photap * pup.psfexpand,
                                     pup.skyin  * pup.psfexpand,
                                     pup.skyout * pup.psfexpand,
                                     skyfrac, pup.expand, pup.skymed)

        # Fraction of the PSF contained in the aperture:
        pup.psffrac += pup.psfskylev * psfnappix
        pup.fp.aplev /= pup.psffrac
        pup.fp.aperr /= pup.psffrac

        pt.msg(1, f'Aperture contains {pup.psffrac:.4f} of the PSF.', pup.log)

    # Make some plots:
    pp.rawflux(
        pup.fp.aplev, pup.fp.aperr, pup.fp.phase, pup.fp.good,
        pup.folder, units=str(pup.units))
    pp.background(
        pup.fp.skylev, pup.fp.phase, pup.fp.good, pup.folder, str(pup.units))

    # Print time stamp, save, and close:
    pt.msg(1,
        f"\nFinished {pup.centering} photometry ({time.ctime()}).\n"
        f"Output folder: '{pup.folder}/'.\n", pup.log)
    io.save(pup)


def calc_photom(pup, data, uncert, mask, start, end,
    aplev, aperr, nappix, skylev, skyerr, nskypix, nskyideal,
    status, good, mute=True):
    """
    Edit (fix) this docstring:
    Medium level routine that performs aperture photometry.
    Each thread from the main routine (photometry) will run do_aphot once.
    do_aphot stores the values in the shared memory arrays.
    """
    # Initialize a Timer to report progress (use first Process):
    if start == 0:
        #clock = t.Timer(pup.inst.nframes,
        #                progress=np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1.1]))
        pass

    y, x = pup.fp.y, pup.fp.x

    # Recalculate star and end indexes. Care not to go out of bounds:
    end = np.amin([end, pup.inst.nframes])

    for i in range(start, end):
        if pup.fp.good[i]:
            # Calculate aperture photometry:
            aperture_photometry = ph.aphot(
                data[i], uncert[i], mask[i], y[i], x[i],
                pup.photap,  pup.skyin,   pup.skyout,
                pup.skyfrac, pup.expand, pup.skymed)

            aplev[i], aperr[i], nappix[i], skylev[i], skyerr[i], \
                nskypix[i], nskyideal[i], status[i] = aperture_photometry
            good[i] = status[i]==0  # good flag

            # Print to screen only if one core:
            if pup.ncpu == 1 and not mute:
                pt.msg(1,
                    f'\nframe = {i:11d}  pos ={pup.fp.pos[i]:3d}  '
                    f'status ={status[i]:3d}  good ={good[i]:3d}\n'
                    f'aplev = {aplev[i]:11.3f}  skylev = {skylev[i]:7.3f}  '
                    f'nappix    = {nappix[i]:7.2f}\n'
                    f'aperr = {aperr[i]:11.3f}  skyerr = {skyerr[i]:7.3f}  '
                    f'nskypix   = {nskypix[i]:7.2f}\n'
                    f'y     = {y[i]:11.3f}  x      = {x[i]:7.3f}  '
                    f'nskyideal = {nskyideal[i]:7.2f}\n', pup.log)

        #perc = 100.0*(i+1.0)/np.sum(pup.inst.nframes)
        #hms = clock.hms_left(np.sum(pup.nimpos[0:pos]) + i)
        #print("progress: {:6.2f}%  -  Remaining time (h:m:s): {}.".
        #      format(perc, hms))

        if start == 0:
            #clock.check(pos*end + i, name=pup.folder)
            pass
