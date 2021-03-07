# Copyright (c) 2018 Patricio Cubillos and contributors.
# puppies is open-source software under the MIT license (see LICENSE).

__all__ = ["photom"]

import os
import time
import shutil
import copy
import configparser
import numpy as np
import multiprocessing as mp

#import timer       as t
#import psf_fit     as pf
#import optphot     as op

from .. import tools as pt
from .. import io    as io
from .. import image as im
from .. import plots as pp
from .. import photometry as ph


def update(pup, cfile):
  """
  Update user parameters for photometry from configuration file,
  and check that all necessary values are well defined.
  """
  # Extract inputs:
  config = configparser.ConfigParser()
  config.optionxform=str
  config.read([cfile])
  args = dict(config.items("PUPPIES"))

  pup.inputs.update(args)
  pt.msg(1, "Updated user parameters: {}".format(list(args.keys())))

  # Set defaults for photometry parameters:
  pup.inputs.setdefault("ncpu",      "1")
  pup.inputs.setdefault("skyfrac",   "0.0")
  pup.inputs.setdefault("skymed",    "False")
  pup.inputs.setdefault("expand",    "1")
  pup.inputs.setdefault("psfexpand", "1")
  pup.inputs.setdefault("otrim",     "10")

  # Required inputs:
  if "photap" not in pup.inputs.keys():
    pt.error("Missing 'photap' user input.")
  if "skyin" not in pup.inputs.keys():
    pt.error("Missing 'skyin' user input.")
  if "skyout" not in pup.inputs.keys():
    pt.error("Missing 'skyout' user input.")

  # FINDME: do no override if values are defaults
  pup.photap    = pt.parray(pup.inputs["photap"])
  pup.skyin     = pt.parray(pup.inputs["skyin"], float)
  pup.skyout    = pt.parray(pup.inputs["skyout"], float)
  pup.ncpu      = int(  pup.inputs["ncpu"])
  pup.skyfrac   = float(pup.inputs["skyfrac"])
  pup.skymed    = bool( pup.inputs["skymed"])
  pup.expand    = int(  pup.inputs["expand"])
  pup.psfexpand = int(  pup.inputs["psfexpand"])
  pup.otrim     = int(  pup.inputs["otrim"])

  # Check aperture photometry is a positive float or 'optimal':
  for i in np.arange(len(pup.photap)):
    try:
      pup.photap[i] = float(pup.photap[i])
      if pup.photap[i] <= 0.0:
        pt.error("Invalid photap aperture: {:.2f}.".format(pup.photap[i]))
    except ValueError:
      if pup.photap[i] != "optimal":
        pt.error("Invalid photap entry: '{:s}'.".format(pup.photap[i]))
  # Check sky annuli inputs have same length than photap:
  if len(pup.skyin) != len(pup.photap):
    if len(pup.skyin) != 1:
      pt.error("'skyin' and 'photap' have inconsistent lengths.")
    pup.skyin = np.repeat(pup.skyin, len(pup.photap))
  if len(pup.skyout) != len(pup.photap):
    if len(pup.skyout) != 1:
      pt.error("'skyout' and 'photap' have inconsistent lengths.")
    pup.skyout = np.repeat(pup.skyout, len(pup.photap))


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
  data   = io.load(pup.datafile,   "data")
  uncert = io.load(pup.uncertfile, "uncert")
  mask   = io.load(pup.maskfile,   "mask")

  # Pre-processing:
  if cfile is not None:
    update(pup, cfile)

  nruns = len(pup.photap)
  # Loop over each run:
  for i in np.arange(nruns):
    # Make a copy of the event:
    puppy = copy.copy(pup)
    puppy.photap = pup.photap[i]
    puppy.skyin  = pup.skyin [i]
    puppy.skyout = pup.skyout[i]

    # Folder suffix when necessary:
    folder = ""
    if puppy.photap != 'optimal':
      folder += 'aper{:03.0f}{:02.0f}{:02.0f}'.format(puppy.photap*100,
                                            puppy.skyin, puppy.skyout)
    else:
      folder = puppy.photap
    #folder += "_{:s}".format(puppy.pcfname)

    # Move into photometry folder:
    puppy.folder = "{:s}/{:s}".format(puppy.folder, folder)
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
    photom = "aperture {:.2f}".format(pup.photap)
  else:
    photom = pup.photap

  # Copy, update, and reopen logfile:
  shutil.copy(pup.logfile, pup.folder)
  pup.logfile = "{:s}/{:s}.log".format(pup.folder, pup.ID)
  pup.log = open(pup.logfile, "a")
  pt.msg(1, "\n\n{:s}\nStarting {:s} photometry  ({:s})\n\n".
            format(70*":", photom, time.ctime()), pup.log)

  # Copy photom.pcf in photdir
  #pcf.make_file("photom.pcf")

  nframes = pup.inst.nframes
  # Aperture photometry:
  if pup.photap != "optimal":
    # Multiprocess set up:
    aplev     = mp.Array("d", np.zeros(nframes))  # aperture flux
    aperr     = mp.Array("d", np.zeros(nframes))  # aperture error
    nappix    = mp.Array("d", np.zeros(nframes))  # number of aperture pixels
    skylev    = mp.Array("d", np.zeros(nframes))  # sky level
    skyerr    = mp.Array("d", np.zeros(nframes))  # sky error
    nskypix   = mp.Array("d", np.zeros(nframes))  # number of sky pixels
    nskyideal = mp.Array("d", np.zeros(nframes))  # ideal number of sky pixels
    status    = mp.Array("i", np.zeros(nframes, int))   # apphot return status
    good      = mp.Array("b", np.zeros(nframes, bool))  # good photometry flag
    # FINDME: Move this allocation out of the if?

    # Size of chunk of data each core will process:
    chunksize = int(nframes/pup.ncpu + 1)
    pt.msg(1, "Number of parallel CPUs for photometry: {:d}."
               .format(pup.ncpu), pup.log)

    # Start Muti Procecess:
    processes = []
    for n in np.arange(pup.ncpu):
      start =  n    * chunksize
      end   = (n+1) * chunksize
      proc = mp.Process(target=calc_photom, args=(pup, data, uncert, mask,
                           start, end, aplev, aperr, nappix, skylev, skyerr,
                           nskypix, nskyideal, status, good))
      processes.append(proc)
      proc.start()
    # Make sure all processes finish their work:
    for n in np.arange(pup.ncpu):
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
    x = np.indices(pshape)
    #clock = t.Timer(np.sum(pup.nimpos),
    #                progress=np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1.1]))

    for i in np.arange(nframes):
      # Integer part of center of subimage:
      cen = np.rint([pup.fp.y[i], pup.fp.x[i]])
      # Center in the trimed image:
      loc = (pup.otrim, pup.otrim)
      # Do the trim:
      img, msk, err = im.trim(pup.data[i], cen, loc, mask=pup.mask[i],
                              uncert=uncert[i])

      # Center of star in the subimage:
      ctr = (pup.fp.y[i]-cen[0]+pup.otrim,
             pup.fp.x[i]-cen[1]+pup.otrim)
      # Make profile:
      # Index of the position in the supersampled PSF:
      pix = pf.pos2index(ctr, pup.expand)
      profile, pctr = pf.make_psf_binning(pup.psfim, pshape, pup.expand,
                                          [pix[0], pix[1], 1.0, 0.0],
                                          pup.psfctr, subpsf)

      # Subtract the sky level:
      img -= pup.fp.skylev[i]
      # optimal photometry calculation:
      immean, imerr, good = op.optphot(img, profile, var=err**2.0, mask=msk)
      # FINDME: Am I not fitting the sky at the same time? I dont like this

      pup.fp.aplev [i] = immean
      pup.fp.aperr [i] = imerr
      pup.fp.skylev[i] = pup.fp.skylev[i]
      pup.fp.good  [i] = good

      # Report progress:
      #clock.check(np.sum(pup.nimpos[0:pos]) + i, name=pup.folder)

  # Print results into the log:
  for i in np.arange(nframes):
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

    pt.msg(1, 'Aperture contains {:.4f} of the PSF.'.format(pup.psffrac),
           pup.log)

  # Make some plots:
  pp.rawflux(pup.fp.aplev, pup.fp.aperr, pup.fp.phase, pup.fp.good,
             pup.folder, units=str(pup.units))
  pp.background(pup.fp.skylev, pup.fp.phase, pup.fp.good, pup.folder,
                str(pup.units))

  # Print time stamp, save, and close:
  pt.msg(1, "\nFinished {:s} photometry  ({:s}).\nOutput folder: '{:s}/'.\n".
                format(pup.centering, time.ctime(), pup.folder), pup.log)
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

  for i in np.arange(start, end):
    if pup.fp.good[i]:
      # Calculate aperture photometry:
      aplev  [i], aperr  [i], nappix   [i], skylev[i], \
       skyerr[i], nskypix[i], nskyideal[i], status[i] = \
                ph.aphot(data[i], uncert[i], mask[i], y[i], x[i],
                         pup.photap,  pup.skyin,   pup.skyout,
                         pup.skyfrac, pup.expand, pup.skymed)
      good[i] = status[i]==0  # good flag

      # Print to screen only if one core:
      if pup.ncpu == 1 and not mute:
        pt.msg(1, '\nframe = {:11d}  pos ={:3d}  status ={:3d}  good ={:3d}\n'
          'aplev = {:11.3f}  skylev = {:7.3f}  nappix    = {:7.2f}\n'
          'aperr = {:11.3f}  skyerr = {:7.3f}  nskypix   = {:7.2f}\n'
          'y     = {:11.3f}  x      = {:7.3f}  nskyideal = {:7.2f}\n'
           .format(i, pup.fp.pos[i], status[i], good[i], aplev[i], skylev[i],
                   nappix[i], aperr[i], skyerr[i], nskypix[i],
                   y[i], x[i], nskyideal[i]), pup.log)

    perc = 100.0*(i+1.0)/np.sum(pup.inst.nframes)
    #hms = clock.hms_left(np.sum(pup.nimpos[0:pos]) + i)
    #print("progress: {:6.2f}%  -  Remaining time (h:m:s): {}.".
    #      format(perc, hms))

    if start == 0:
      #clock.check(pos*end + i, name=pup.folder)
      pass
