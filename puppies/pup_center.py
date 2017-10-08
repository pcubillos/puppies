import os
import time
import shutil
import copy
import configparser
import multiprocessing as mp
import numpy as np

import astropy.io.fits as fits

from . import tools  as pt
from . import io     as io
from . import stats  as ps
from . import center as pc

topdir = os.path.realpath(os.path.dirname(__file__) + "/../")

"""
CENTERING WORKFLOW
------------------
This beautiful piece of code consists of four sections:
- driver: Parse inputs, loop over centering methods
- centering: center PSF and mean image, launch multi-process over frames
- loop: loop over frames to run center.center()
"""

def update(pup, cfile):
  # Extract inputs:
  config = configparser.ConfigParser()
  config.optionxform=str
  config.read([cfile])
  args = dict(config.items("PUPPIES"))

  pup.inputs.update(args)
  pt.msg(1, "Updated user parameters: {}".format(list(args.keys())))

  # Set defaults for centering parameters:
  pup.inputs.setdefault("ncpu",     "1")
  pup.inputs.setdefault("ctrim",    "8")
  pup.inputs.setdefault("fitbg",    "True")
  pup.inputs.setdefault("cweights", "False")
  pup.inputs.setdefault("aradius",  "0")
  pup.inputs.setdefault("asize",    "0")
  pup.inputs.setdefault("psftrim",  "0")
  pup.inputs.setdefault("psfarad",  "0")
  pup.inputs.setdefault("psfasize", "0")
  pup.inputs.setdefault("expand",   "0")

  # check all necessary inputs are provided:
  if "centering" not in pup.inputs.keys():
    pt.error("Missing 'centering' user input.")

  pup.centering = pt.parray(pup.inputs["centering"])
  pup.ncpu      = int( pup.inputs["ncpu"])
  pup.ctrim     = int( pup.inputs["ctrim"])
  pup.cweights  = bool(pup.inputs["cweights"])
  pup.fitbg     = bool(pup.inputs["fitbg"])
  pup.aradius   = int( pup.inputs["aradius"])
  pup.asize     = int( pup.inputs["asize"])
  pup.psftrim   = int( pup.inputs["psftrim"])
  pup.psfarad   = int( pup.inputs["psfarad"])
  pup.psfasize  = int( pup.inputs["psfasize"])
  pup.expand    = int( pup.inputs["expand"])

  if "lag" in pup.centering:
    if pup.aradius == 0 or pup.asize == 0:
      pt.error("Missing 'aradius' or 'asize' least-asymmetry user inputs.")
    if os.path.isfile(pup.psf):
      if pup.psfarad == 0 or pup.psfasize == 0:
        pt.error("Missing 'psfaradius' or 'psfasize' least-asym. user inputs.")

  if "psffit" in pup.centering:
    if pup.expand == 0:
      pt.error("Missing 'expand' centering user input.")


def driver(pup, cfile=None):
  """
  Read the config file.
  Update pup.
  Loop over each centering.
  """
  # Current (badpix) pup folder:
  cwd = pup.folder

  # Load data:
  pup.datafile = pup.data
  pup.uncdfile = pup.uncd
  pup.maskfile = pup.mask
  pup.data = io.load(pup.data, "data") * pup.fluxunits
  pup.uncd = io.load(pup.uncd, "uncd") * pup.fluxunits
  pup.mask = io.load(pup.mask, "mask")

  # Pre-processing:
  if cfile is not None:
    update(pup, cfile)

  nruns = len(pup.centering)
  # Loop over each run:
  for i in np.arange(nruns):
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
    centering(puppy)
  #return list_of_puppies_for_next_step


def centering(pup):
  """
  Target position finding.
  """
  # Copy, update, and reopen logfile:
  shutil.copy(pup.logfile, pup.folder)
  pup.logfile = "{:s}/{:s}.log".format(pup.folder, pup.ID)
  pup.log = open(pup.logfile, "a")
  pt.msg(1, "\n\n{:s}\nStarting {:s} centering  ({:s})\n\n".
            format(70*":", pup.centering, time.ctime()), pup.log)

  # Check least asym parameters work:
  if pup.centering in ['lac', 'lag']:
    if pup.ctrim < (pup.aradius + pup.asize) and pup.ctrim != 0:
      pup.ctrim = pup.aradius + pup.asize + 3
      pt.msg(1, 'Trim radius is too small, changed to: {:d}'.format(pup.ctrim),
             pup.log)
    if pup.psftrim < (pup.psfarad + pup.psfasize) and pup.psftrim !=0:
      pup.psftrim = pup.psfarad + pup.psfasize + 3
      pt.msg(1, 'PSF Trim radius is too small, changed to: {:d}'.
                 format(pup.psftrim), pup.log)

  # PSF Centering:
  if os.path.isfile(pup.psf):
    pup.psfim, psfheader = fits.getdata(pup.psf, header=True,
                                        ignore_missing_end=True)
    # Guess of the center of the PSF (center of psfim)
    psfctrguess = np.asarray(np.shape(pup.psfim))/2
    # Find center of PSF:
    pup.psfctr, extra = pc.center(pup.centering, pup.psfim, psfctrguess,
                               pup.psftrim, pup.psfarad, pup.psfasize)
    pt.msg(1, 'PSF center found at: {}'.format(pup.psfctr), pup.log)
  else:
    pup.psfim  = None
    pup.psfctr = None
    pt.msg(1, 'No PSF supplied.', pup.log)

  # Find center of the mean Image:
  pup.targpos = np.zeros((2, pup.inst.npos))
  for pos in np.arange(pup.inst.npos):
    meanim = pup.meanim[:,:,pos]
    guess  = pup.srcest[:, pos]
    targpos, extra = pc.center(pup.centering, meanim, guess, pup.ctrim,
                      pup.aradius, pup.asize, fitbg=pup.fitbg,
                      psf=pup.psfim, psfctr=pup.psfctr, expand=pup.expand)
    pup.targpos[:,pos] = targpos
  pt.msg(1, "Center position(s) of the mean Image(s):\n{}".
            format(pup.targpos.T), pup.log)

  # Multy Process set up:
  x       = mp.Array("d", np.zeros(pup.inst.npos * pup.inst.maxnimpos))
  y       = mp.Array("d", np.zeros(pup.inst.npos * pup.inst.maxnimpos))
  flux    = mp.Array("d", np.zeros(pup.inst.npos * pup.inst.maxnimpos))
  sky     = mp.Array("d", np.zeros(pup.inst.npos * pup.inst.maxnimpos))
  goodfit = mp.Array("d", np.zeros(pup.inst.npos * pup.inst.maxnimpos))

  # Size of chunk of data each core will process:
  chunksize = int(pup.inst.maxnimpos/pup.ncpu + 1)
  pt.msg(1, "Number of parallel CPUs: {:d}.".format(pup.ncpu), pup.log)

  # Start Muti Procecess: ::::::::::::::::::::::::::::::::::::::
  processes = []
  for n in np.arange(pup.ncpu):
    start =  n    * chunksize # Starting index to process
    end   = (n+1) * chunksize # Ending   index to process
    proc = mp.Process(target=center, args=(pup, start, end, #centermask,
                         x, y, flux, sky, goodfit))
    processes.append(proc)
    proc.start()
  # Make sure all processes finish their work:
  for n in np.arange(pup.ncpu):
    processes[n].join()

  # Put the results in the object. I need to reshape them:
  pup.fp.x         = np.array(x   ).reshape(pup.inst.npos, pup.inst.maxnimpos)
  pup.fp.y         = np.array(y   ).reshape(pup.inst.npos, pup.inst.maxnimpos)
  # If PSF fit:
  if pup.centering in ["ipf", "bpf"]: 
    pup.fp.flux    = np.array(flux).reshape(pup.inst.npos, pup.inst.maxnimpos)
    pup.fp.psfsky  = np.array(sky ).reshape(pup.inst.npos, pup.inst.maxnimpos)
    pup.fp.goodfit = np.array(goodfit).reshape(pup.inst.npos,pup.inst.maxnimpos)
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # Distance to closest pixel center:
  pup.fp.r = np.sqrt((pup.fp.x % 1.0 - 0.5)**2.0 + 
                     (pup.fp.y % 1.0 - 0.5)**2.0 )

  # Delete data arrays:
  pup.data = pup.datafile
  pup.uncd = pup.uncdfile
  pup.mask = pup.maskfile
  del(pup.datafile, pup.uncdfile, pup.maskfile)
  # Print time stamp, save, and close:
  pt.msg(1, "\nFinished {:s} centering  ({:s}).\nOutput folder: '{:s}/'.\n".
                format(pup.centering, time.ctime(), pup.folder), pup.log)
  io.save(pup)


def center(pup, start, end, x, y, flux, sky, goodfit):
  """
  Doc Me!
  """
  # Initialize a Timer to report progress:
  if start == 0:  # Only for the fisrt chunk
    #clock = t.Timer(pup.npos*end,
    #                progress=np.array([0.05, 0.1, 0.2, 0.3, 0.4,  0.5,
    #                                   0.6,  0.7, 0.8, 0.9, 0.99, 1.1]))
    pass
  data = pup.data

  # Finally, do the centering:
  for pos in np.arange(pup.inst.npos):
    # Recalculate start/end, Care not to go out of bounds:
    #start = np.amin([start, pup.nimpos[pos]]) # is this necessary?
    end   = np.amin([end,   pup.nimpos[pos]])
    for i in np.arange(start, end):
      # Index in the share memory arrays:
      ind = pos * pup.inst.maxnimpos + i
      try:
        if pup.cweights:   # weight by uncertainties in fitting?
          uncd = pup.uncd[i,:,:,pos]
        else:
          uncd = None
        # Do the centering:
        position, extra = pc.center(pup.centering, data[i,:,:,pos],
                               pup.targpos[:,pos], pup.ctrim,
                               pup.aradius, pup.asize,
                               pup.mask[i,:,:,pos],
                               uncd, fitbg=pup.fitbg,
                               expand=pup.expand,
                               psf=pup.psfim, psfctr=pup.psfctr)
        y[ind], x[ind] = position
        if pup.centering in ["ipf", "bpf"]:
          flux[ind] = extra[0]
          sky [ind] = extra[1]
          # FINDME: define some criterion for good/bad fit.
          goodfit[ind] = 1
      except:
        y[ind], x[ind] = pup.targpos[:, pos]
        flux[ind], sky[ind] = 0.0, 0.0
        goodfit[ind] = 0
        pt.msg(1, "Centering failed in image {:5d}, position: {:2d}".
                   format(i, pos), pup.log)

      if start == 0: 
        #print("{}/{}".format(i,end))
        # Report progress:
        #clock.check(pos*end + im, name=pup.folder)
        pass
