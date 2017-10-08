import os
import sys
import shutil
import time
import numpy as np

import astropy.io.fits as fits
import astropy.units   as u

from . import tools as pt
from . import io    as io
from . import stats as ps

topdir = os.path.realpath(os.path.dirname(__file__) + "/../")


def badpix(pup):
  """
  Bad pixel masking (so far for Spitzer pup).
  """
  # Move to output folder:
  os.chdir(pup.folder)
  pup.folder += "/badpix"
  if not os.path.exists(pup.folder):
    os.mkdir(pup.folder)
  os.chdir(pup.folder)

  # Copy, update, and reopen logfile:
  shutil.copy(pup.logfile, pup.folder)
  pup.logfile = pup.folder + "/" + pup.ID + ".log"
  pup.log = open(pup.logfile, "a")
  pt.msg(1, "\n\n{:s}\nStarting bad pixel masking  ({:s})\n\n".
            format(70*":", time.ctime()), pup.log)

  # Julian observation date
  #pup.fp.juldat = pup.jdjf80 + pup.fp.time / 86400.0

  # ::::::::::::::::::::::: FLUX CONVERSION :::::::::::::::::::::::::::::
  # Do we want flux (uJy/pix) or surface brightness (MJy/sr) units?  If
  # doing photometry, convert to flux.  Since we care about relative
  # numbers, it doesn't really matter.

  # Convert from surface brightness (MJy/sr) to flux units (uJy/pix)
  inst = pup.inst
  pup.fluxunits = inst.bunit*inst.posscl[0]*inst.posscl[1]
  if pup.units is not None:
    pup.fluxunits = pup.fluxunits.to(u.Unit(pup.units))

  data   = io.load(pup.data, "data")   * pup.fluxunits
  uncd   = io.load(pup.uncd, "uncd")   * pup.fluxunits
  bdmskd = io.load(pup.uncd, "bdmskd")

  # Mean Background Estimate, from zodi model
  pup.estbg = ( np.mean(pup.fp.zodi[pup.fp.exist]) +
                np.mean(pup.fp.ism [pup.fp.exist]) +
                np.mean(pup.fp.cib [pup.fp.exist]) ) * pup.fluxunits

  # Get permanent bad pixel mask.
  if not os.path.exists(pup.inst.pmaskfile[0]):
    pt.error('Permanent Bad pixel mask not found!', pup.log)
  else:
    hdu = fits.open(pup.inst.pmaskfile[0])
    if hdu[0].header['bitpix'] == -32:  # if data type is float
      hdu[0].scale(type='int16')        # cast it down to int16
    pup.pmask = hdu[0].data

  # IRS FIX: IRS data contains the blue peak subarray while its
  #          pmask contains the whole array
  if pup.inst.chan == 5:
    pup.pmask = pup.pmask[3:59,86:127]  # (Hard coding)

  # Do NOT define sigma, we have a different scheme for finding baddies
  # adds Spitzer rejects: fp.nsstrej  &  our rejects: fp.nsigrej
  pt.msg(1, "Apply Spitzer bad pixel mask.", pup.log)
  pup.mask = badmask(data, uncd, pup.pmask,  pup.inst.pcrit,
                     bdmskd, pup.inst.dcrit, pup.fp, pup.nimpos)

  ## User rejected pixels:
  #if pup.userrej != None:
  #  for i in np.arange(np.shape(pup.userrej)[0]):
  #    pup.mask[:, pup.userrej[i,0], pup.userrej[i,1], :] = 0
  #  pup.fp.userrej = np.sum(np.sum(1-pup.mask, axis=1), axis=1)
  #  pup.fp.userrej = np.transpose(pup.fp.userrej) - pup.fp.nsstrej
  #else:
  #  pup.fp.userrej = np.zeros((pup.npos, pup.maxnimpos))

  # Sigma rejection:
  pt.msg(1, "Sigma Rejection.", pup.log)
  chunkbad(data, uncd, pup.mask, pup.nimpos, 
           pup.sigma, pup.schunk, pup.fp) #, pup.inst.nscyc)

  pt.msg(1, 'Compute mean frame and mean sky.', pup.log)
  # Mean image per pixel per pos:
  totdat = np.nansum(data*pup.mask, axis=0)
  totmsk = np.nansum(pup.mask,      axis=0)
  totmsk[totmsk == 0] = 1.0  # Avoid dividing by zero
  pup.meanim = totdat / totmsk

  # Approximate sky for every image (median of an image):
  pup.fp.medsky = np.zeros((inst.npos, inst.maxnimpos)) * data.unit
  for pos in np.arange(inst.npos):
    for i in np.arange(pup.nimpos[pos]):
      pup.fp.medsky[pos, i] = np.median(data[i,:,:,pos][pup.mask[i,:,:,pos]])
  # Apply sigma rejection on medsky here?

  # Print time elapsed and close log:
  pt.msg(1, "\nFinished bad-pixel masking  ({:s}).\nOutput folder: '{:s}'.\n".
         format(time.ctime(), pup.folder), pup.log)
  io.save(pup)


def badmask(data, uncd, pmask, pcrit, dmask, dcrit, fp, nimpos):
  """
  Generate a bad pixel mask from Spitzer time-series photometry data.

  Parameters
  ----------
  data: 4D float ndarray
     Array of shape (maxnimpos,ny,nx,npos) , where nx and ny
     are the image dimensions, maxnimpos is the maximum number
     of images in the largest set, and npos is the number of
     sets (or 'positions').
  uncd: 4D float ndarray
     Uncertainties of data.
  pmask: 2D integer ndarray
     Permanent bad pixel mask for the instrument.
  pcrit: Integer
     A bitmask indicating which bits in Pmask are critical
     problems for which we should flag a bad pixel.
  dmask: 4D integer ndarray
     Per-frame bad pixel masks for the dataset.  Same
     shape as Data, maybe different type.
  dcrit: Integer
     Per-frame bad pixel mask frame.
  fp: 2D ndarray
     Per-frame parameters, of shape (npos, maxnimpos)
  nimpos:  1D ndarray
     zero-based index of the last valid image in each position
     of data.
  sigma:  1D ndarray
     Successive sigma-rejection threshholds, passed to sigrej.
     If not defined, the data check is skipped, so that another
     routine can be used for that step.  Still allocates the
     array and initializes it with pre-flagged bad pixels (NaNs
     and prior masks).

  Return
  ------
  mask: 4D bool ndarray
    Good pixel mask of data pixels (True=good, False=bad).
  """

  # sizes
  maxnimpos, ny, nx, npos = np.shape(data)

  # allocate bad pixel mask
  mask = np.zeros((maxnimpos, ny, nx, npos), bool)

  # flag existing frames as good
  for pos in np.arange(npos):
    mask[0:nimpos[pos], :, :, pos] = True

  # Permanently-bad pixel mask:
  pmaskcrit = pmask & pcrit
  if np.amax(pmaskcrit) > 0:
    mask &= np.expand_dims(pmaskcrit==0, 2)

  # Per-frame bad pixel mask:
  dmaskcrit = dmask & dcrit
  mask &= (dmaskcrit==0)

  # Flag NaNs in the data and uncertainties:
  finite = np.isfinite(data) & np.isfinite(uncd)
  mask &= finite

  # Spitzer-rejected bad pixels:
  #fp.nsstrej = np.sum(np.sum(1 - mask, axis=1), axis=1)
  fp.nsstrej = np.sum(1-mask, axis=(1,2))

  fp.nsstrej = np.transpose(fp.nsstrej)
  return mask


def chunkbad(data, uncd, mask, nimpos, sigma, szchunk, fp):
  """
  Sigma-rejection bad pixels for Spitzer Space Telescope data.

  For each block of szchunk images, it does sigma rejection
  in each pixel position flagging any outliers and recording them
  appropriately in the NSIGREJ column of FP.

  Parameters
  ----------
  data: 4D float ndarray
     The data to be analyzed. Of shape [nim,nx,ny,npos], nx and ny
     are the image dimensions, nim is the maximum number of images
     in the largest set, and npos is the number of sets (or 'positions').
  uncd: 4D float ndarray
     Uncertainties of corresponding points in data, same shape as data.
  mask: 4D byte ndarray
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
  maxnimpos, ny, nx, npos = np.shape(data)

  # Our rejects
  nsigrej = np.zeros((maxnimpos, npos))

  # High value constant
  highval = 1e100 * data.unit

  # Initialize a Timer to report progress
  #clock = t.Timer(np.sum(nimpos)*1.0/szchunk)

  for pos in np.arange(npos):
    # number of chunks:
    nchunk = int(np.ceil(nimpos[pos]/szchunk))
    # Aliases:
    dat = data[:,:,:,pos]
    msk = mask[:,:,:,pos]

    for chunk in np.arange(nchunk):
      # Chunk boundaries:
      start = chunk*szchunk
      end   = np.clip((chunk+1)*szchunk, 0, nimpos[pos])

      # Do sigma rejection within the chunk, replace that and pre-masked stuff
      # it's ok if there are pre-flagged data in the sigma rejection input
      keepmsk = ps.sigrej(dat.value[start:end], sigma, msk[start:end])
      keepmsk &= msk[start:end]

      # Number of rejected pixels:
      nsigrej[start:end, pos] += np.sum(1-keepmsk, axis=(1,2))

      # Flag bad data:
      reject = np.where(keepmsk == 0)  # location in chunk
      reject[0][:] += start               # location in data

      # Update mask and uncertainty arrays:
      #uncd[:,:,:,pos][reject] = highval
      mask[:,:,:,pos][reject] = False

      # Report progress:
      #clock.check(np.sum(nimpos[0:pos])*1.0/szchunk + chunk)

  # Nsigrej now holds all the bad pixels.  How many are new?
  fp.nsigrej = nsigrej.T - fp.nsstrej #- fp.userrej
