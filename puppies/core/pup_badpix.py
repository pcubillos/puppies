# Copyright (c) 2021-2024 Patricio Cubillos
# puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

__all__ = [
    'badpix',
]

import os
import shutil
import time

import astropy.io.fits as fits
import astropy.units as u
import numpy as np

from .. import io as io
from .. import stats as ps
from .. import tools as pt


def badpix(pup):
    """
    Bad pixel masking (so far for Spitzer pup).
    """
    # Move to output folder:
    with pt.cd(pup.folder):
        pup.folder /= 'badpix'
        if not os.path.exists(pup.folder):
            os.mkdir(pup.folder)

    # Copy, update, and reopen logfile:
    shutil.copy(pup.logfile, pup.folder)
    pup.logfile = pup.folder / f"{pup.ID}.log"
    log = pup.log = pt.Log(pup.logfile, append=True)
    log.msg(f"\n\n{log.sep}\nStarting bad pixel masking ({time.ctime()})\n\n")

    # Julian observation date
    #pup.fp.juldat = pup.jdjf80 + pup.fp.time / 86400.0

    # Do we want flux (uJy/pix) or surface brightness (MJy/sr) units?  If
    # doing photometry, convert to flux.  Since we care about relative
    # numbers, it doesn't really matter.

    # Convert from surface brightness (MJy/sr) to flux units (uJy/pix)
    inst = pup.inst
    pup.fluxunits = inst.bunit*inst.posscl[0]*inst.posscl[1]
    if pup.units is not None:
        pup.fluxunits = pup.fluxunits.to(u.Unit(pup.units))

    data = io.load(pup.datafile, "data") * pup.fluxunits
    uncert = io.load(pup.uncertfile, "uncert") * pup.fluxunits
    bdmskd = io.load(pup.bdmskdfile, "bdmskd")

    # Mean Background Estimate, from zodi model
    pup.estbg = pup.fluxunits * (
        np.mean(pup.fp.zodi) +
        np.mean(pup.fp.ism)  +
        np.mean(pup.fp.cib)
    )

    # Get permanent bad pixel mask.
    if not os.path.exists(pup.inst.pmaskfile[0]):
        log.error('Permanent Bad pixel mask not found!')
    else:
        hdu = fits.open(pup.inst.pmaskfile[0])
        if hdu[0].header['bitpix'] == -32:  # if data type is float
            hdu[0].scale(type='int16') # cast it down to int16
        pup.pmask = hdu[0].data

    # IRS hard-coded fix: IRS data contains the blue peak subarray while its
    # pmask contains the whole array
    if pup.inst.chan == 5:
        pup.pmask = pup.pmask[3:59,86:127]

    # Do NOT define sigma, we have a different scheme for finding baddies
    # adds Spitzer rejects: fp.nsstrej  &  our rejects: fp.nsigrej
    log.msg("Apply bad pixel masks.")
    pup.mask = badmask(
        data, uncert, pup.pmask,  pup.inst.pcrit,
        bdmskd, pup.inst.dcrit, pup.fp, pup.nimpos,
    )

    ## User rejected pixels:
    #if pup.userrej != None:
    #  for i in range(np.shape(pup.userrej)[0]):
    #    pup.mask[:, pup.userrej[i,0], pup.userrej[i,1], :] = 0
    #  pup.fp.userrej = np.sum(np.sum(1-pup.mask, axis=1), axis=1)
    #  pup.fp.userrej = np.transpose(pup.fp.userrej) - pup.fp.nsstrej
    #else:
    #  pup.fp.userrej = np.zeros((pup.npos, pup.maxnimpos))

    # Sigma rejection:
    log.msg("Sigma Rejection.")
    chunkbad(data, uncert, pup.mask, pup.nimpos, pup.sigma, pup.schunk, pup.fp)

    log.msg('Compute mean frame and mean sky.')
    # Mean image per pixel per position:
    pup.meanim = np.zeros((pup.inst.npos, pup.inst.ny, pup.inst.nx))
    for pos in range(pup.inst.npos):
        ipos = pup.fp.pos == pos
        totdat = np.nansum((data*pup.mask)[ipos], axis=0)
        totmsk = np.nansum(pup.mask[ipos], axis=0)
        totmsk[totmsk == 0] = 1.0  # Avoid dividing by zero
        pup.meanim[pos] = totdat / totmsk

    # Approximate sky for every image (median of an image):
    pup.fp.medsky = np.zeros(inst.nframes) * data.unit
    for i in range(inst.nframes):
        pup.fp.medsky[i] = np.median(data[i][pup.mask[i]])
    # Apply sigma rejection on medsky here?

    # Print time elapsed and close log:
    log.msg(
        f"\nFinished bad-pixel masking  ({time.ctime()})."
        f"\nOutput folder: '{pup.folder}'.\n")
    io.save(pup)


def badmask(data, uncert, pmask, pcrit, dmask, dcrit, fp, nimpos):
    """
    Generate a bad pixel mask from Spitzer time-series photometry data.

    Parameters
    ----------
    data: 3D float ndarray
        Array of shape (nframes,ny,nx), where nx and ny are the image
        dimensions, nframes is the number of images.
    uncert: 3D float ndarray
        Uncertainties of data.
    pmask: 2D integer ndarray
        Permanent bad pixel mask for the instrument.
    pcrit: Integer
        A bitmask indicating which bits in Pmask are critical
        problems for which we should flag a bad pixel.
    dmask: 3D integer ndarray
        Per-frame bad pixel masks for the dataset.  Same
        shape as Data, maybe different type.
    dcrit: Integer
        Per-frame bad pixel mask frame.
    fp: 2D ndarray
        Per-frame parameters, of shape (npos, maxnimpos)
    nimpos:  1D ndarray
        zero-based index of the last valid image in each position of data.
    sigma:  1D ndarray
        Successive sigma-rejection threshholds, passed to sigrej.
        If not defined, the data check is skipped, so that another
        routine can be used for that step.  Still allocates the
        array and initializes it with pre-flagged bad pixels (NaNs
        and prior masks).

    Return
    ------
    mask: 3D bool ndarray
        Good pixel mask of data pixels (True=good, False=bad).
    """

    # sizes
    nframes, ny, nx = np.shape(data)

    # allocate bad pixel mask
    mask = np.ones((nframes, ny, nx), bool)

    # Permanently-bad pixel mask:
    pmaskcrit = pmask & pcrit
    if np.amax(pmaskcrit) > 0:
        mask &= pmaskcrit==0

    # Per-frame bad pixel mask:
    dmaskcrit = dmask & dcrit
    mask &= dmaskcrit==0

    # Flag NaNs in the data and uncertainties:
    finite = np.isfinite(data) & np.isfinite(uncert)
    mask &= finite

    # Spitzer-rejected bad pixels per frame:
    fp.nsstrej = np.sum(1-mask, axis=(1,2))
    return mask


def chunkbad(data, uncert, mask, nimpos, sigma, szchunk, fp):
    """
    Sigma-rejection bad pixels for Spitzer Space Telescope data.

    For each block of szchunk images, it does sigma rejection
    in each pixel position flagging any outliers and recording them
    appropriately in the NSIGREJ column of FP.

    Parameters
    ----------
    data: 3D float ndarray
        The data to be analyzed. Of shape [nim,ny,nx], nx and ny
        are the image dimensions, nim is the maximum number of images.
    uncert: 3D float ndarray
        Uncertainties of corresponding points in data, same shape as data.
    mask: 3D byte ndarray
        Good-pixel mask (True=good, False=bad), same shape as data.
    nimpos: 1D integer array
        The number of good images at each photometry position.
    sigma: 1D float array
        Passed to sigrej (see sigrej documentation).
    szchunk: Integer
        Number of images in a processing chunk, usually equal to the
        number in a subarray-mode readoutset.
    fp: frame parameters variable.
    """
    # Sizes
    nframes, ny, nx = np.shape(data)
    npos = len(nimpos)

    # Our rejects
    nsigrej = np.zeros(nframes)

    for pos in range(npos):
        ipos = fp.pos == pos
        # number of chunks:
        nchunk = int(np.ceil(nimpos[pos]/szchunk))
        # Aliases:
        dat = data[ipos]
        msk = mask[ipos]

        for chunk in range(nchunk):
            # Chunk boundaries:
            start = chunk*szchunk
            end = np.clip((chunk+1)*szchunk, 0, nimpos[pos])

            # Do sigma rejection within chunk, replace that and pre-masked stuff
            # it's ok if there are pre-flagged data in the sigma rejection input
            keepmsk = ps.sigrej(dat.value[start:end], sigma, msk[start:end])
            keepmsk &= msk[start:end]

            # Number of rejected pixels:
            nsigrej[ipos][start:end] += np.sum(1-keepmsk, axis=(1,2))

            # Flag bad data:
            reject = np.where(keepmsk == 0)  # location in chunk
            reject[0][:] += start            # location in data

            # Update mask and uncertainty arrays:
            mask[ipos][reject] = False

    # Nsigrej now holds all the bad pixels.  How many are new?
    fp.nsigrej = nsigrej - fp.nsstrej #- fp.userrej
