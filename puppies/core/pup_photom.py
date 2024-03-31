# Copyright (c) 2021-2024 Patricio Cubillos
# puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

__all__ = [
    'photom',
]

import os
import time
import shutil
import copy
import multiprocessing as mp

import numpy as np

from .. import image as im
from .. import io as io
from .. import photometry as ph
from .. import plots as pp
from .. import tools as pt


def photom(pup, cfile=None):
    """
    Load the event.
    Read config file.
    Launch a thread for each centering run.
    """
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
    for i in range(nruns):
        # Make a copy of the event:
        puppy = copy.copy(pup)
        puppy.photap = pup.photap[i]
        puppy.skyin  = pup.skyin[i]
        puppy.skyout = pup.skyout[i]

        # Folder suffix when necessary:
        if puppy.photap != 'optimal':
            ap_name = f'{puppy.photap*100:03.0f}'
            sky_in = f'{puppy.skyin:02.0f}'
            sky_out = f'{puppy.skyout:02.0f}'
            folder = f'aper{ap_name}{sky_in}{sky_out}'
        else:
            folder = puppy.photap

        # Move into photometry folder:
        puppy.folder = f'{puppy.folder}/{folder}'
        with pt.cd(pup.folder):
            if not os.path.exists(puppy.folder):
                os.mkdir(puppy.folder)

        # Launch the thread:
        photometry(puppy, data, uncert, mask)

    #return list_of_puppies_for_next_step


def photometry(pup, data, uncert, mask):
    """
    Doc me.
    """
    fp = pup.fp

    if isinstance(pup.photap, float):
        photom = f"aperture {pup.photap:.2f}"
    else:
        photom = pup.photap

    # Copy, update, and reopen logfile:
    shutil.copy(pup.logfile, pup.folder)
    pup.logfile = f"{pup.folder}/{pup.ID}.log"
    log = pup.log = pt.Log(pup.logfile, append=True)
    log.msg(
        f"\n\n{log.sep}\nStarting {photom} photometry  ({time.ctime()})\n\n"
    )

    nframes = pup.inst.nframes
    # Aperture photometry:
    if pup.photap != "optimal":
        aplev = mp.Array("d", np.zeros(nframes))  # aperture flux
        aperr = mp.Array("d", np.zeros(nframes))  # aperture error
        nappix = mp.Array("d", np.zeros(nframes))  # number of aperture pixels
        skylev = mp.Array("d", np.zeros(nframes))  # sky level
        skyerr = mp.Array("d", np.zeros(nframes))  # sky error
        nskypix = mp.Array("d", np.zeros(nframes))  # number of sky pixels
        nskyideal = mp.Array("d", np.zeros(nframes))  # ideal sky pixels
        status = mp.Array("i", np.zeros(nframes, int))   # apphot return status
        good = mp.Array("b", np.zeros(nframes, bool))  # good photometry flag
        # FINDME: Move this allocation out of the if?

        # Size of chunk of data each core will process:
        chunksize = int(nframes/pup.ncpu + 1)
        log.msg(f"Number of parallel CPUs for photometry: {pup.ncpu}.")

        pup_data = [
            pup.fp,
            pup.inst.nframes,
            pup.photap,
            pup.skyin,
            pup.skyout,
            pup.skyfrac,
            pup.expand,
            pup.skymed,
            pup.ncpu,
        ]
        # Start Muti Procecess:
        processes = []
        for n in range(pup.ncpu):
            start =  n    * chunksize
            end   = (n+1) * chunksize
            args = (
                pup_data, data, uncert, mask, start, end, aplev, aperr, nappix,
                skylev, skyerr, nskypix, nskyideal, status, good,
            )
            proc = mp.Process(target=calc_photom, args=args)
            processes.append(proc)
            proc.start()
        for n in range(pup.ncpu):
            processes[n].join()

        # Put the results in the event. I need to reshape them:
        fp.aplev = np.array(aplev)
        fp.aperr = np.array(aperr)
        fp.nappix = np.array(nappix)
        fp.skylev = np.array(skylev)
        fp.skyerr = np.array(skyerr)
        fp.nskypix = np.array(nskypix)
        fp.nskyideal = np.array(nskyideal)
        fp.status = np.array(status)
        fp.goodphot = np.array(good, bool)
        # Overwrite good with goodphot:
        fp.good = np.copy(fp.goodphot)

        # Raw photometry (star + sky flux within the aperture):
        fp.apraw = fp.aplev + fp.skylev*fp.nappix

    elif pup.photap == "optimal":
        # utils for profile construction:
        pshape = np.array([2*pup.otrim+1, 2*pup.otrim+1])
        subpsf = np.zeros(np.asarray(pshape, float)*pup.expand)

        for i in range(nframes):
            # Integer part of center of subimage:
            cen = np.rint([fp.y[i], fp.x[i]])
            # Center in the trimed image:
            loc = (pup.otrim, pup.otrim)
            # Do the trim:
            img, msk, err = im.trim(
                pup.data[i], cen, loc, mask=pup.mask[i], uncert=uncert[i])

            # Center of star in the subimage:
            ctr = (
                fp.y[i]-cen[0]+pup.otrim,
                fp.x[i]-cen[1]+pup.otrim,
            )
            # Make profile:
            # Index of the position in the supersampled PSF:
            pix = ph.position_to_index(ctr, pup.expand)
            profile, pctr = ph.make_psf_binning(
                pup.psfim, pshape, pup.expand,
                [pix[0], pix[1], 1.0, 0.0],
                pup.psfctr, subpsf,
            )

            # Subtract the sky level:
            img -= fp.skylev[i]
            # optimal photometry calculation:
            immean, imerr, good = ph.optimal_photometry(
                img, profile, var=err**2.0, mask=msk,
            )
            # TBD: Does not fit the sky at the same time

            fp.aplev[i] = immean
            fp.aperr[i] = imerr
            fp.skylev[i] = fp.skylev[i]
            fp.good[i] = good

    # Print results into the log:
    for i in range(nframes):
        log.msg(
            f'\nframe = {i:11d}  pos ={fp.pos[i]:3d}  '
            f'status ={fp.status[i]:3d}  good ={fp.good[i]:3d}\n'
            f'aplev = {fp.aplev[i]:11.3f}  skylev = {fp.skylev[i]:7.3f}  '
            f'nappix    = {fp.nappix[i]:7.2f}\n'
            f'aperr = {fp.aperr[i]:11.3f}  skyerr = {fp.skyerr[i]:7.3f}  '
            f'nskypix   = {fp.nskypix[i]:7.2f}\n'
            f'y     = {fp.y[i]:11.3f}  x      = {fp.x[i]:7.3f}  '
            f'nskyideal = {fp.nskyideal[i]:7.2f}\n'
        )

    if pup.centering in ["bpf"]:
        pup.ispsf = False

    if os.path.isfile(pup.psf) and isinstance(pup.photap, float):
        # PSF aperture correction:
        log.msg('Calculating PSF aperture.')
        pup.psfim = pup.psfim.astype(np.float64)

        imerr = np.ones(np.shape(pup.psfim))
        imask = np.ones(np.shape(pup.psfim), bool)
        skyfrac = 0.1
        ap_photometry = ph.aphot(
            pup.psfim, imerr, imask,
            pup.psfctr[0],
            pup.psfctr[1],
            pup.photap * pup.psfexpand,
            pup.skyin  * pup.psfexpand,
            pup.skyout * pup.psfexpand,
            skyfrac, pup.expand, pup.skymed,
        )
        pup.psffrac = ap_photometry[0]
        aperr = ap_photometry[1]
        psfnappix = ap_photometry[2]
        pup.psfskylev = ap_photometry[3]
        skyerr = ap_photometry[4]
        #psfnskypix = ap_photometry[5]
        #psfnskyideal = ap_photometry[6]
        pup.psfstatus = ap_photometry[7]

        # Fraction of the PSF contained in the aperture:
        pup.psffrac += pup.psfskylev * psfnappix
        pup.fp.aplev /= pup.psffrac
        pup.fp.aperr /= pup.psffrac

        log.msg(f'Aperture contains {pup.psffrac:.4f} of the PSF.')

    # Make some plots:
    pp.rawflux(
        pup.fp.aplev, pup.fp.aperr, pup.fp.phase, pup.fp.good,
        pup.folder, units=str(pup.units),
    )
    pp.background(
        pup.fp.skylev, pup.fp.phase, pup.fp.good, pup.folder, str(pup.units),
    )

    # Print time stamp, save, and close:
    log.msg(
        f"\nFinished {pup.centering} photometry ({time.ctime()}).\n"
        f"Output folder: '{pup.folder}/'.\n"
    )
    io.save(pup)


def calc_photom(
        pup_data, data, uncert, mask, start, end,
        aplev, aperr, nappix, skylev, skyerr, nskypix, nskyideal,
        status, good, mute=True,
    ):
    """
    Medium level routine that performs aperture photometry.
    Each thread from the main routine (photometry) will run do_aphot once.
    do_aphot stores the values in the shared memory arrays.
    """
    fp, nframes, photap, skyin, skyout, skyfrac, expand, skymed, \
        ncpu = pup_data
    y, x = fp.y, fp.x

    # Recalculate star and end indexes. Care not to go out of bounds:
    end = np.amin([end, nframes])

    for i in range(start, end):
        if fp.good[i]:
            # Calculate aperture photometry:
            aperture_photometry = ph.aphot(
                data[i], uncert[i], mask[i], y[i], x[i],
                photap, skyin, skyout,
                skyfrac, expand, skymed,
            )
            aplev[i] = aperture_photometry[0]
            aperr[i] = aperture_photometry[1]
            nappix[i] = aperture_photometry[2]
            skylev[i] = aperture_photometry[3]
            skyerr[i] = aperture_photometry[4]
            nskypix[i] = aperture_photometry[5]
            nskyideal[i] = aperture_photometry[6]
            status[i] = aperture_photometry[7]
            good[i] = status[i]==0

            # Print to screen only if one core:
            if ncpu == 1 and not mute:
                print(
                    f'\nframe = {i:11d}  pos ={fp.pos[i]:3d}  '
                    f'status ={status[i]:3d}  good ={good[i]:3d}\n'
                    f'aplev = {aplev[i]:11.3f}  skylev = {skylev[i]:7.3f}  '
                    f'nappix    = {nappix[i]:7.2f}\n'
                    f'aperr = {aperr[i]:11.3f}  skyerr = {skyerr[i]:7.3f}  '
                    f'nskypix   = {nskypix[i]:7.2f}\n'
                    f'y     = {y[i]:11.3f}  x      = {x[i]:7.3f}  '
                    f'nskyideal = {nskyideal[i]:7.2f}\n'
                )

