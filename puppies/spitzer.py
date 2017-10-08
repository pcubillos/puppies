import time

import os
import sys
import re
import numpy as np
import astropy.io.fits   as fits
import matplotlib.pyplot as plt

import astropy.coordinates as coord
import astropy.constants   as ac
import astropy.units       as u
import astropy.wcs         as wcs

from . import tools as pt
from . import io    as io

topdir = os.path.realpath(os.path.dirname(__file__) + "/../")


class Pup():
  """
  Pup class for a Spitzer data set.
  """
  def __init__(self, args):
    self.initpars(args)
    self.calc()
    self.read()
    self.check()
    io.save(self)


  def initpars(self, args):
    # Put user inputs into the Pup object:
    self.inputs = inputs = args
    # Event:
    self.planetname = inputs["planetname"]
    self.ID         = inputs["ID"]

    # Create output folder:
    self.root = inputs["root"]
    if self.root == "default":
      self.root = os.getcwd()
    if not os.path.exists(self.root):
      pt.error("Output root folder does not exists ('{:s}').".format(self.root))

    self.folder = self.root + "/" + self.ID
    if not os.path.exists(self.folder):
      os.mkdir(self.folder)
    os.chdir(self.folder)

    # Make file to store running log:
    self.logfile = "{:s}/{:s}.log".format(self.folder, self.ID)
    self.log = open(self.logfile, "w")
    pt.msg(1, "\nStarting new Puppies project in '{:s}'.".
                  format(self.folder), self.log)

    # Parse Parameters:
    ra  = inputs["ra"].split()
    dec = inputs["dec"].split()
    self.ra  = coord.Angle(ra[0],  unit=u.Unit(ra[1]))
    self.dec = coord.Angle(dec[0], unit=u.Unit(dec[1]))

    # Parameter   uncertainty:
    self.rstar,   self.urstar   = pt.getpar(inputs["rstar"])
    self.Zstar,   self.uZstar   = pt.getpar(inputs["Zstar"])
    self.tstar,   self.utstar   = pt.getpar(inputs["tstar"])
    self.logg,    self.ulogg    = pt.getpar(inputs["logg"])
    self.rplanet, self.urplanet = pt.getpar(inputs["rplanet"])
    self.smaxis,  self.usmaxis  = pt.getpar(inputs["smaxis"])
    self.incl,    self.uincl    = pt.getpar(inputs["incl"])
    self.ephtime, self.uephtime = pt.getpar(inputs["ephtime"])
    self.period,  self.uperiod  = pt.getpar(inputs["period"])
    self.T14,     self.uT14     = pt.getpar(inputs["T14"])

    self.rprs2  = (self.rplanet/self.rstar)**2
    self.urprs2 = 2*self.rprs2 * np.sqrt((self.urplanet/self.rplanet)**2 +
                                         (self.urstar  /self.rstar  )**2 )

    self.units = u.Unit(inputs["units"])

    # Instrument-specific:
    self.inst = Instrument(inputs["instrument"])
    self.inst.npos    = int(inputs["npos"])
    self.inst.nnod    = int(inputs["nnod"])
    self.inst.aorname = pt.parray(inputs["aorname"])
    self.inst.naor    = len(self.inst.aorname)
    self.inst.datadir = inputs["data"]

    # Ancilliary files:
    self.horizons = inputs["horizons"]
    if self.horizons == "default":
      self.horizons = topdir + "/inputs/spitzer/all_spitzer.vec"

    self.kurucz   = inputs["kurucz"]

    self.filter   = inputs["filter"]
    if self.filter == "default":
      self.filter = "{:s}/inputs/spitzer/filter/{:s}".format(
                                topdir, self.inst.filter_def[self.inst.chan-1])

    self.psf      = inputs["psf"]
    if self.psf == "default":
      self.psf = "{:s}/inputs/spitzer/psf/{:s}".format(
                                topdir, self.inst.psf_def[self.inst.chan-1])

    self.inst.pmaskfile = []
    for i in np.arange(self.inst.naor):
      self.inst.pmaskfile.append("{:s}/r{:s}/{:s}/cal/{:s}".
        format(self.inst.datadir, self.inst.aorname[i], self.inst.channel,
               inputs["pmaskfile"]))

    # Sigma rejection:
    self.schunk = int(inputs["schunk"])
    self.sigma  = pt.parray(inputs["sigma"], float)


  def calc(self):
    inst = self.inst
    inst.nexpid  = np.zeros(inst.naor, np.int)

    # compile patterns: lines ending with each suffix
    bcdpattern    = re.compile("(.+" + inst.bcdsuf    + ")\n")
    bdmskpattern  = re.compile("(.+" + inst.bdmsksuf  + ")\n")
    bdmsk2pattern = re.compile("(.+" + inst.bdmsksuf2 + ")\n")

    # Make list of files in each AOR:
    inst.bcdfiles = []
    for aor in np.arange(inst.naor):
      bcddir = "{:s}/r{:s}/{:s}/bcd/".format(inst.datadir, inst.aorname[aor],
                                             inst.channel)
      frameslist = os.listdir(bcddir)
      framesstring = '\n'.join(frameslist) + '\n'

      # find the data files
      bcdfiles = bcdpattern.findall(framesstring)
      # and sort them
      inst.bcdfiles.append(sorted(bcdfiles))

      # find bdmask suffix:
      if bdmskpattern.findall(framesstring) != []:
        inst.masksuf = inst.bdmsksuf
      elif bdmsk2pattern.findall(framesstring) != []:
        inst.masksuf = inst.bdmsksuf2
      else:
        pt.error("No mask files found.", self.log)

      # get first index of exposition ID, number of expID, and ndcenum
      #                    expid      dcenum     pipev
      first = re.search("_([0-9]{4})_([0-9]{4})_([0-9])",inst.bcdfiles[-1][0])
      last  = re.search("_([0-9]{4})_([0-9]{4})_([0-9])",inst.bcdfiles[-1][-1])

      inst.expadj      = int(first.group(1))
      inst.nexpid[aor] = int(last.group(1)) + 1 - inst.expadj
      inst.ndcenum     = int(last.group(2)) + 1
      inst.pipev       = int(last.group(3))

    # pick a random image, not the first
    data, head = fits.getdata(bcddir + inst.bcdfiles[-1][-2], header=True)

    # Number of characters in the header:
    inst.hsize = len(head.tostring())

    # data size
    shape = data.shape
    if data.ndim >= 3:
      inst.nz, inst.ny, inst.nx = shape
    else:
      inst.nz = 1
      inst.ny, inst.nx = shape

    # Number of small, medium and big cycles:
    if inst.name.startswith("irac"):
      inst.nbcyc = 1
      inst.nmcyc = np.sum(inst.nexpid)
      inst.nscyc = inst.ndcenum if inst.nz == 1 else inst.nz
    elif inst.name == 'irs':
      inst.nbcyc = np.sum(inst.nexpid) // inst.nnod
      inst.nmcyc = inst.ndcenum
      inst.nscyc = 1
    elif inst.name == 'mips':
      inst.nbcyc = np.sum(inst.nexpid) // inst.nnod
      inst.nmcyc = (inst.ndcenum - 1) // 7
      inst.nscyc = 7

    # Max. number of images per position:
    if inst.name == 'mips':
      inst.maxnimpos = inst.nbcyc * (inst.nmcyc + 1)
    else:
      inst.maxnimpos = np.sum(inst.nexpid) * inst.ndcenum * inst.nz // inst.nnod

    # Header info:
    try:
      inst.framtime = head['FRAMTIME']   # interval between exposure starts
    except:
      inst.framtime = 0.0
    try:
      inst.exptime  = head['EXPTIME']    # effective exposure time
    except:
      inst.exptime  = None
    try:
      inst.gain     = head['GAIN']       # e/DN conversion
    except:
      inst.gain     = None

    inst.sscver   = head["CREATOR"]   # Spitzer pipeline
    inst.bunit    = u.Unit(head['BUNIT'])  # Units of image data
    inst.fluxconv = head['FLUXCONV']  # Flux Conv factor (MJy/Str per DN/sec)
    # Pixel scale:
    inst.posscl = [np.abs(head['PXSCAL2']), np.abs(head['PXSCAL1'])]*u.arcsec

    if inst.chan != head['CHNLNUM']:  # Spitzer photometry channel
      pt.error('Unexpected photometry channel!', self.log)


  def read(self):
    """
    Read a set of IRAC AORS, (or IRAC Subarray AORS), sorting by dither
    position, if any.
    """
    inst = self.inst
    # Allocate space for returned arrays:
    headerdtype = 'S'+str(inst.hsize)
    head   = np.zeros((inst.maxnimpos//inst.nz, inst.npos), headerdtype)
    data   = np.zeros((inst.maxnimpos, inst.ny, inst.nx, inst.npos), float)
    uncd   = np.zeros((inst.maxnimpos, inst.ny, inst.nx, inst.npos), float)
    bdmskd = np.zeros((inst.maxnimpos, inst.ny, inst.nx, inst.npos), int)
    brmskd = np.zeros((inst.maxnimpos, inst.ny, inst.nx, inst.npos), int)

    # FP contains values per frame (duh!):
    fp = FrameParameters(inst.npos, inst.maxnimpos)
    nimpos = np.zeros(inst.npos, np.int)

    # dictionary to get position in MIPS
    mirind = {1929.:0, 2149.5:1, 1907.5:2, 2128.:3,
              1886.:4, 2106.5:5, 1864.5:6}

    # Write to log first line
    pt.msg(1, "\nEvent data:\n  aor  expid  dcenum   pos", self.log)

    # pattern to find     expid      dcenum
    pattern = re.compile("_([0-9]{4})_([0-9]{4})_")

    # Obtain data
    for aor in np.arange(inst.naor):
      bcddir = "{:s}/r{:s}/{:s}/bcd/".format(inst.datadir, inst.aorname[aor],
                                             inst.channel)
      bcd   = inst.bcdfiles[aor]
      for i in np.arange(len(bcd)):
        bcdfile = os.path.realpath(bcddir + bcd[i])
        # Read data
        try:
          dataf, bcdhead = fits.getdata(bcdfile, header=True)
        except: # If a file doesn't exist, skip to next file.
          pt.warning(1, "BCD file not found: {:s}".format(bcdfile), self.log)
          continue

        try: # Read uncertainity and mask files
          # Replace suffix in bcd file to get the corresponding file.
          uncfile = re.sub(inst.bcdsuf, inst.buncsuf, bcdfile)
          uncf    = fits.getdata(uncfile)
          mskfile = re.sub(inst.bcdsuf, inst.masksuf, bcdfile)
          bdmskf  = fits.getdata(mskfile)
        except:
          pass

        try: # Mips
          brmskfile = re.sub(inst.bcdsuf, inst.brmsksuf, bcdfile)
          brmskf    = fits.getdata(brmskfile)
        except:
          brmskf    = -np.ones((inst.nz, inst.ny, inst.nx), np.int)

        # Obtain expid and dcenum
        index = pattern.search(bcd[i])
        expid  = int(index.group(1))
        dcenum = int(index.group(2))

        # Do I really need this?
        if np.size(bdmskf) == 1:
          bdmskf = -np.ones((inst.nz, inst.ny, inst.nx), np.int)
        if np.size(brmskf) == 1:
          brmskf = -np.ones((inst.nz, inst.ny, inst.nx), np.int)

        # Find dither position
        try:
          pos = bcdhead['DITHPOS'] - 1
        except:
          pos = 0  # No dither position in stare data
        if inst.name == 'irs':
          pos = expid % inst.npos
        elif inst.name == 'mips':
          nod = expid % inst.nnod
          pos = nod * inst.nscyc + mirind[bcdhead['CSM_PRED']]

        be = nimpos[pos]           # begining
        en = nimpos[pos] + inst.nz # end

        # Store data
        data  [be:en, :, :, pos] = dataf.reshape( (inst.nz, inst.ny, inst.nx))
        uncd  [be:en, :, :, pos] = uncf.reshape(  (inst.nz, inst.ny, inst.nx))
        bdmskd[be:en, :, :, pos] = bdmskf.reshape((inst.nz, inst.ny, inst.nx))
        brmskd[be:en, :, :, pos] = brmskf.reshape((inst.nz, inst.ny, inst.nx))
        # All the single numbers per frame that we care about
        fp.frmobs[pos, be:en] = np.sum(nimpos) + np.arange(inst.nz)
        fp.pos   [pos, be:en] = pos
        fp.aor   [pos, be:en] = aor
        fp.expid [pos, be:en] = expid
        fp.dce   [pos, be:en] = dcenum
        fp.subarn[pos, be:en] = np.arange(inst.nz)
        fp.time  [pos, be:en] = (bcdhead['UTCS_OBS'] +
                                 inst.framtime * (0.5 + np.arange(inst.nz)))
        fp.pxscl1[pos, be:en] = np.abs(bcdhead['PXSCAL1'])
        fp.pxscl2[pos, be:en] = np.abs(bcdhead['PXSCAL2'])
        try:
          fp.zodi    [pos, be:en] = bcdhead['ZODY_EST']
          fp.ism     [pos, be:en] = bcdhead['ISM_EST']
          fp.cib     [pos, be:en] = bcdhead['CIB_EST']
          fp.afpat2b [pos, be:en] = bcdhead['AFPAT2B']
          fp.afpat2e [pos, be:en] = bcdhead['AFPAT2E']
          fp.ashtempe[pos, be:en] = bcdhead['ASHTEMPE'] + 273.0
          fp.atctempe[pos, be:en] = bcdhead['ATCTEMPE'] + 273.0
          fp.acetempe[pos, be:en] = bcdhead['ACETEMPE'] + 273.0
          fp.apdtempe[pos, be:en] = bcdhead['APDTEMPE'] + 273.0
          fp.acatmp1e[pos, be:en] = bcdhead['ACATMP1E']
          fp.acatmp2e[pos, be:en] = bcdhead['ACATMP2E']
          fp.acatmp3e[pos, be:en] = bcdhead['ACATMP3E']
          fp.acatmp4e[pos, be:en] = bcdhead['ACATMP4E']
          fp.acatmp5e[pos, be:en] = bcdhead['ACATMP5E']
          fp.acatmp6e[pos, be:en] = bcdhead['ACATMP6E']
          fp.acatmp7e[pos, be:en] = bcdhead['ACATMP7E']
          fp.acatmp8e[pos, be:en] = bcdhead['ACATMP8E']
        except:
          pass

        try:
          fp.acatmp5e[pos, be:en] = bcdhead['CMD_T_24']
          fp.acatmp6e[pos, be:en] = bcdhead['AD24TMPA']
          fp.acatmp6e[pos, be:en] = bcdhead['AD24TMPB']
          fp.acatmp5e[pos, be:en] = bcdhead['ACSMMTMP']
          fp.acatmp6e[pos, be:en] = bcdhead['ACEBOXTM'] + 273.0
        except:
          pass

        # Store filename
        fp.filename[pos, be:en] = os.path.realpath(bcddir + bcd[i])

        # Store header
        head[np.int(nimpos[pos]/inst.nz), pos] = str(bcdhead)

        # Header position of the star:
        bcdhead["NAXIS"] = 2
        WCS = wcs.WCS(bcdhead, naxis=2)
        pix = WCS.wcs_world2pix([[self.ra.deg, self.dec.deg]], 0)
        fp.headx[pos, be:en] = pix[0,0]
        fp.heady[pos, be:en] = pix[0,1]
        nimpos[pos] += inst.nz

        # Print to log and screen:
        pt.msg(1, "{:4d}{:7d}{:7d}{:7d}".format(aor, expid, dcenum, pos),
               self.log)

    # where there exist data
    for pos in np.arange(inst.npos):
      fp.exist[pos, 0:nimpos[pos]] = True

    for pos in np.arange(inst.npos):
      fp.im[pos, 0:nimpos[pos]] = np.arange(nimpos[pos], dtype=np.double)

    # Set cycle number, visit within obs. set, frame within visit:
    if inst.name == "mips":
      fp.cycpos = np.trunc(fp.frmobs / (2*inst.ndcenum))
      fp.visobs = np.trunc(fp.frmobs / inst.ndcenum)
      fp.frmvis = np.trunc(fp.frmobs % inst.ndcenum)
    else:
      fp.cycpos = np.trunc(fp.frmobs // (inst.npos * inst.nmcyc * inst.nz))
      fp.visobs = np.trunc(fp.frmobs // (            inst.nmcyc * inst.nz))
      fp.frmvis = fp.im % (inst.nmcyc * inst.nz)

    # Update event:
    self.data   = data
    self.uncd   = uncd
    self.bdmskd = bdmskd
    self.brmskd = brmskd
    self.head   = head
    self.mask   = ""
    self.fp     = fp
    self.nimpos = nimpos


  def check(self):
    inst = self.inst
    # Source estimated position
    self.srcest = np.zeros((2, inst.npos))
    for p in np.arange(inst.npos):
      self.srcest[0,p] = np.mean(self.fp.heady[p,:self.nimpos[p]])
      self.srcest[1,p] = np.mean(self.fp.headx[p,:self.nimpos[p]])

    # Plot a reference image
    image = np.zeros((inst.ny, inst.nx))
    for pos in np.arange(inst.npos):
      image += self.data[0, :, :, pos]

    plt.figure(101, (8,6))
    plt.clf()
    plt.imshow(image, interpolation='nearest', origin='ll', cmap=plt.cm.viridis)
    plt.plot(self.srcest[1,:], self.srcest[0,:],'k+', ms=12, mew=2)
    plt.xlim(-0.5, inst.nx-0.5)
    plt.ylim(-0.5, inst.ny-0.5)
    plt.title(self.ID + ' reference image')
    plt.colorbar()
    plt.savefig(self.ID + "_sample-frame.png")

    # Throw a warning if the source estimate position lies outside of
    # the image.
    warning = False
    if (np.any(self.srcest[1,:] < 0) or np.any(self.srcest[1,:] > inst.nx) or
        np.any(self.srcest[0,:] < 0) or np.any(self.srcest[0,:] > inst.ny) ):
      pt.warning(1, "Source RA-DEC position lies out of bounds.", self.log)

    # Write to log
    pt.msg(1, "\nSummary:\nTarget:     {:s}\nEvent name: {:s}".
           format(self.planetname, self.ID), self.log)
    pt.msg(1, "Spitzer pipeline version: {:s}".format(inst.sscver),
           self.log)
    pt.msg(1, "AOR files: {}\nExposures per AOR: {}".
           format(inst.aorname, inst.nexpid), self.log)
    pt.msg(1, "Number of target positions: {:d}\n"
           "Target guess position (pixels):\n {:}".
           format(inst.npos, self.srcest), self.log)
    pt.msg(1, "Frames per position: {}\nRead a total of {:d} frames.\n".
           format(self.nimpos, np.sum(self.nimpos)), self.log)

    # Report files not found:
    print("Ancil Files:")
    if not os.path.isfile(inst.pmaskfile[0]):
      pt.warning(1, "Pmask file not found ('{:s}').".format(inst.pmaskfile[0]),
                 self.log)
    else:
      pt.msg(1, "Pmask file: '{:s}'".format(inst.pmaskfile[0]), self.log, 2)

    if not os.path.isfile(self.horizons):
      pt.warning(1, "Horizon file not found ('{:s}').".format(self.horizons),
                 self.log)
    else:
      pt.msg(1, "Horizon file: '{:s}'".format(self.horizons), self.log, 2)

    if not os.path.isfile(self.kurucz):
      pt.warning(1, "Kurucz file not found ('{:s}').".format(self.kurucz),
                 self.log)
    else:
      pt.msg(1, "Kurucz file: '{:s}'".format(self.kurucz), self.log, 2)

    if not os.path.isfile(self.filter):
      pt.warning(1, "Filter file not found ('{:s}').".format(self.filter),
                 self.log)
    else:
      pt.msg(1, "Filter file: '{:s}'".format(self.filter), self.log, 2)

    if not os.path.isfile(self.psf):
      pt.warning(1, "PSF file not found ('{:s}').".format(self.psf), self.log)
    else:
      pt.msg(1, "PSF file: '{:s}'".format(self.psf), self.log, 2)

    if self.inst.exptime is None:
      pt.warning(1, "Exposure time undefined.", self.log)
    if self.inst.gain is None:
      pt.warning(1, "Gain undefined.", self.log)


class Instrument:
  def __init__(self, inst):
    self.name = inst
    if self.name == "mips":
      self.chan    = 6
      self.channel = "/ch1"
      self.prefix  = "M"
      self.wavel = 24.0 * u.Unit("micron")
    elif self.name == "irs":
      self.chan    = 5
      self.channel = "/ch0"
      self.prefix  = "S"
      self.wavel = 16.0 * u.Unit("micron")
    elif self.name.startswith("irac"):
      self.chan = int(self.name[-1])
      self.channel = "/ch{:d}".format(self.chan)
      self.prefix = "I"
      self.wavel = [0, 3.6, 4.5, 5.8, 8.0][self.chan] * u.Unit("micron")
    else:
      print("Wrong instrument name.")

    # Frequency calculated from wavelength
    self.freq = (ac.c/self.wavel).decompose()

    if   self.name.startswith("irac"):
      self.bcdsuf   = '_bcd.fits'     # bcd image (basic calibrated data)
      self.buncsuf  = '_bunc.fits'    # bcd uncertainties
      self.bdmsksuf  = '_bimsk.fits'  # bcd outlier mask
      self.bdmsksuf2 = '_bdmsk.fits'  # bcd outlier mask
      self.brmsksuf = '_brmsk.fits'   # bcd outlier mask
    elif self.name == "irs":
      self.bcdsuf   = '_bcdb.fits'    # bcd image (basic calibrated data)
      self.buncsuf  = '_uncb.fits'    # bcd uncertainties
      self.bdmsksuf = '_b_msk.fits'   # bcd outlier mask
      self.bdmsksuf2 = '_xxxx.fits'   # inelegant solution
      self.brmsksuf = '_mskb.fits'    # bcd outlier mask
    elif self.name == "mips":
      self.bcdsuf   = '_bcd.fits'     # bcd image (basic calibrated data)
      self.buncsuf  = '_bunc.fits'    # bcd uncertainties
      self.bdmsksuf  = '_bbmsk.fits'  # bcd outlier mask
      self.bdmsksuf2 = '_xxxx.fits'   # inelegant solution
      self.brmsksuf = '_brmsk.fits'   # bcd outlier mask

    self.msaicsuf = '_msaic.fits'     # pbcd mosaic image
    self.msuncsuf = '_msunc.fits'     # pbcd mosaic uncertainties
    self.mscovsuf = '_mscov.fits'     # pbcd mosaic coverage (number of images)
    self.irsasuf  = '_irsa.tbl'       # list of 2MASS sources in the field
    self.pmasksuf = '_pmask.fits'     # pointing-refinement-corrected keywords

    # Critical mask flags
    self.pcrit   = np.long(65535)  # in pmask (permanent bad-pixel
                                   # mask, IRACDH2.0, T4.1)
    if self.chan < 6:
      # in dmask (per-frame bad-pixel mask, IRACDH2.0, T4.2) added bit
      # 4 (decimal 16) since uncerts are high and flux is low in top
      # row, which has this flag
      self.dcrit   = np.int_(32560)
    else:
      self.dcrit   = np.long(65024)

    # Default ancilliary files:
    self.filter_def = ["irac1_filter.dat", "irac2_filter.dat",
                       "irac3_filter.dat", "irac3_filter.dat",
                       "irs-blue_filter.dat", "mips-24um_filter.dat"]
    self.psf_def = ["IRAC.1.PRF.5X.070312.fits", "IRAC.1.PRF.5X.070312.fits",
                    "IRAC.1.PRF.5X.070312.fits", "IRAC.1.PRF.5X.070312.fits",
                    "IRS_BPUI_PSF.fits"]


class FrameParameters:
  """
  class holder of the frame parameters.
  """
  def __init__(self, npos, maxnimpos):
    self.frmobs   = np.zeros((npos,maxnimpos), int)  # Frame number
    self.pos      = np.zeros((npos,maxnimpos), int)  # Position number
    self.aor      = np.zeros((npos,maxnimpos), int)  # AOR number
    self.expid    = np.zeros((npos,maxnimpos), int)  # Exposure ID
    self.dce      = np.zeros((npos,maxnimpos), int)  # Data Collection Event
    self.subarn   = np.zeros((npos,maxnimpos), int)  # Subarray frame number
    self.exist    = np.zeros((npos,maxnimpos), bool) # frame tags in fp
    self.im       = np.zeros((npos,maxnimpos), int)  # Frame within position
    self.cycpos   = np.zeros((npos,maxnimpos), int)
    self.visobs   = np.zeros((npos,maxnimpos), int)
    self.frmvis   = np.zeros((npos,maxnimpos), int)
    self.time     = np.zeros((npos,maxnimpos))  # Frame mid-time (s) J2000.0
    self.zodi     = np.zeros((npos,maxnimpos))  # Zodiacal light estimate
    self.ism      = np.zeros((npos,maxnimpos))  # interstellar medium estimate
    self.cib      = np.zeros((npos,maxnimpos))  # Cosmic infrared background
    self.afpat2b  = np.zeros((npos,maxnimpos))  # Temperatures
    self.afpat2e  = np.zeros((npos,maxnimpos))
    self.ashtempe = np.zeros((npos,maxnimpos))
    self.atctempe = np.zeros((npos,maxnimpos))
    self.acetempe = np.zeros((npos,maxnimpos))
    self.apdtempe = np.zeros((npos,maxnimpos))
    self.acatmp1e = np.zeros((npos,maxnimpos))
    self.acatmp2e = np.zeros((npos,maxnimpos))
    self.acatmp3e = np.zeros((npos,maxnimpos))
    self.acatmp4e = np.zeros((npos,maxnimpos))
    self.acatmp5e = np.zeros((npos,maxnimpos))
    self.acatmp6e = np.zeros((npos,maxnimpos))
    self.acatmp7e = np.zeros((npos,maxnimpos))
    self.acatmp8e = np.zeros((npos,maxnimpos))
    # mips frame parameters
    self.cmd_t_24 = np.zeros((npos,maxnimpos))
    self.ad24tmpa = np.zeros((npos,maxnimpos))
    self.ad24tmpb = np.zeros((npos,maxnimpos))
    self.acsmmtmp = np.zeros((npos,maxnimpos))
    self.aceboxtm = np.zeros((npos,maxnimpos))
    self.pxscl2   = np.zeros((npos,maxnimpos))
    self.pxscl1   = np.zeros((npos,maxnimpos))

    self.heady    = np.zeros((npos,maxnimpos))
    self.headx    = np.zeros((npos,maxnimpos))
    self.filename = np.zeros((npos,maxnimpos), dtype='S150')
