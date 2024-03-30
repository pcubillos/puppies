# Copyright (c) 2021-2024 Patricio Cubillos
# puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

__all__ = [
    "Spitzer",
]

import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as ac
import astropy.coordinates as coord
import astropy.io.fits as fits
import astropy.time as at
import astropy.units as u
import astropy.wcs as wcs

from .. import tools as pt
from .. import io as io
from ..tools import ROOT


class Spitzer():
    """
    Pup class for a Spitzer data set.
    """
    def __init__(self, args):
        # Put user inputs into the Pup object:
        self.inputs = inputs = args
        self.planetname = inputs["planetname"]
        self.ID = inputs["ID"]

        # Create output folder:
        self.root = inputs["root"]
        if self.root == "default":
            self.root = os.getcwd()
        root_path = Path(self.root).expanduser().resolve()

        if not root_path.exists():
            pt.error(f"Root folder does not exists: '{self.root}'.")

        self.folder = root_path / self.ID
        if not os.path.exists(self.folder):
            self.folder.mkdir()
        os.chdir(self.folder)

        # Make file to store running log:
        self.logfile = f"{self.folder}/{self.ID}.log"
        self.log = log = pt.Log(self.logfile)
        log.msg(f"\nStarting new Puppies project in '{self.folder}'.")

        # Parse Parameters:
        ra = inputs["ra"].split()
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
        self.period,  self.uperiod  = pt.getpar(inputs["period"])
        self.T14,     self.uT14     = pt.getpar(inputs["T14"])
        # Setting the ephemeris time is a bit more complicated:
        ephemeris = inputs["ephtime"].split()
        self.ephtime = at.Time(
            float(ephemeris[0]), format="jd", scale=ephemeris[3])
        # Correct from HJD to BJD if necessary:
        if ephemeris[2].lower() == "hjd":
            sun = coord.get_body_barycentric(
                'sun', self.ephtime, ephemeris='builtin')
            target = coord.SkyCoord(self.ra, self.dec).cartesian
            self.ephtime += at.TimeDelta(np.sum(target.xyz*sun.xyz)/ac.c)

        self.uephtime = float(ephemeris[1]) # * u.d

        self.rprs2 = (self.rplanet/self.rstar)**2.0
        self.urprs2 = 2*self.rprs2 * np.sqrt(
            (self.urplanet/self.rplanet)**2 +
            (self.urstar/self.rstar)**2
        )

        self.units = u.Unit(inputs["units"])

        # Instrument-specific:
        inst = self.inst = Instrument(inputs["instrument"])
        inst.npos = int(inputs["npos"])
        inst.nnod = int(inputs["nnod"])
        inst.aorname = pt.parray(inputs["aorname"], dtype=str)
        inst.naor = len(self.inst.aorname)
        inst.datadir = inputs["data"]

        # Ancilliary files:
        self.kurucz = inputs["kurucz"]

        self.filter = f"{ROOT}puppies/data/spitzer_filters/{inst.default_filter}"
        if inputs["filter"] != "default":
            self.filter = inputs["filter"]

        self.psf = f"{ROOT}puppies/data/spitzer_psf/{inst.default_psf}"
        if inputs["psf"] != "default":
            self.psf = inputs["psf"]

        inst.pmaskfile = []
        for aor in inst.aorname:
            inst.pmaskfile.append(
                f'{inst.datadir}/r{aor}/{inst.channel}/'
                f'cal/{inputs["pmaskfile"]}'
        )

        # Sigma rejection:
        self.schunk = int(inputs["schunk"])
        self.sigma = pt.parray(inputs["sigma"], float)

        # Now do some calculations:
        inst.nexpid = np.zeros(inst.naor, int)

        # compile patterns: lines ending with each suffix
        bcdpattern = re.compile("(.+" + inst.bcdsuf + ")\n")
        bdmskpattern = re.compile("(.+" + inst.bdmsksuf  + ")\n")
        bdmsk2pattern = re.compile("(.+" + inst.bdmsksuf2 + ")\n")

        # Make list of files in each AOR:
        inst.bcdfiles = []
        # Total number of files (may not be equal to the number of images):
        inst.nfiles = 0
        for aor in range(inst.naor):
            bcddir = f"{inst.datadir}/r{inst.aorname[aor]}/{inst.channel}/bcd/"
            frameslist = os.listdir(bcddir)
            framesstring = '\n'.join(frameslist) + '\n'

            # Find BCD data files and sort them:
            bcdfiles = bcdpattern.findall(framesstring)
            inst.bcdfiles.append(sorted(bcdfiles))
            inst.nfiles += len(inst.bcdfiles[aor])

            # Find bdmask suffix:
            if bdmskpattern.findall(framesstring) != []:
                inst.masksuf = inst.bdmsksuf
            elif bdmsk2pattern.findall(framesstring) != []:
                inst.masksuf = inst.bdmsksuf2
            else:
                inst.masksuf = None
                log.warning("No BCD mask files found")

            first_bcd = inst.bcdfiles[-1][0]
            last_bcd = inst.bcdfiles[-1][-1]
            # Get first index of exposition ID, number of expID, and ndcenum
            #                    expid      dcenum     pipev
            first = re.search("_([0-9]{4})_([0-9]{4})_([0-9])", first_bcd)
            last  = re.search("_([0-9]{4})_([0-9]{4})_([0-9])", last_bcd)

            inst.expadj = int(first.group(1))
            inst.nexpid[aor] = int(last.group(1)) + 1 - inst.expadj
            inst.ndcenum = int(last.group(2)) + 1
            inst.pipev = int(last.group(3))

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
            inst.maxnimpos = \
                np.sum(inst.nexpid) * inst.ndcenum * inst.nz // inst.nnod

        # Total number of images:
        inst.nframes = inst.nfiles * inst.nz

        # interval between exposure starts
        inst.framtime = 0.0
        if 'FRAMTIME' in head:
            inst.framtime = head['FRAMTIME']

        # effective exposure time
        inst.exptime = None
        if 'EXPTIME' in head:
            inst.exptime = head['EXPTIME']

        # e/DN conversion
        inst.gain = None
        if 'GAIN' in head:
            inst.gain = head['GAIN']

        inst.sscver = head["CREATOR"]   # Spitzer pipeline
        inst.bunit = u.Unit(head['BUNIT'])  # Units of image data
        # Flux Conv factor (MJy/Str per DN/sec)
        inst.fluxconv = head['FLUXCONV']
        # Pixel scale:
        inst.posscl = \
            [np.abs(head['PXSCAL2']), np.abs(head['PXSCAL1'])] * u.arcsec

        if inst.chan != head['CHNLNUM']:  # Spitzer photometry channel
            log.error('Unexpected photometry channel.')


        # Read the data now:
        inst = self.inst
        # Allocate space for returned arrays:
        headerdtype = f'S{inst.hsize}'
        head = np.zeros((inst.nframes//inst.nz), headerdtype)
        data = np.zeros((inst.nframes, inst.ny, inst.nx), float)
        uncert = np.zeros((inst.nframes, inst.ny, inst.nx), float)
        bdmskd = np.zeros((inst.nframes, inst.ny, inst.nx), int)
        brmskd = np.zeros((inst.nframes, inst.ny, inst.nx), int)

        # FP contains values per frame (duh!):
        fp = FrameParameters(inst.nframes)
        telem = fp.telemetry
        nimpos = np.zeros(inst.npos, int)
        nframes = 0
        # Dictionary to get position in MIPS:
        mirind = {
            1929.0: 0,
            2149.5: 1,
            1907.5: 2,
            2128.0: 3,
            1886.0: 4,
            2106.5: 5,
            1864.5: 6,
        }

        # Write to log first line:
        log.msg("\nEvent data:\n  aor  expid  dcenum   pos")

        # pattern to find     expid      dcenum
        pattern = re.compile("_([0-9]{4})_([0-9]{4})_")

        x_hcrs = np.zeros(inst.nframes)
        y_hcrs = np.zeros(inst.nframes)
        z_hcrs = np.zeros(inst.nframes)
        time = np.zeros(inst.nframes) * u.d

        # Obtain data
        for aor in range(inst.naor):
            bcddir = f"{inst.datadir}/r{inst.aorname[aor]}/{inst.channel}/bcd/"
            bcd = inst.bcdfiles[aor]
            for i in range(len(bcd)):
                bcdfile = os.path.realpath(bcddir + bcd[i])
                # Read data
                try:
                    dataf, bcdhead = fits.getdata(bcdfile, header=True)
                except: # If a file doesn't exist, skip to next file.
                    log.warning(f"BCD file not found: {bcdfile}")
                    continue

                try: # Read uncertainity and mask files
                    # Replace suffix in bcd file to get the corresponding file.
                    uncfile = re.sub(inst.bcdsuf, inst.buncsuf, bcdfile)
                    uncf = fits.getdata(uncfile)
                    mskfile = re.sub(inst.bcdsuf, inst.masksuf, bcdfile)
                    bdmskf = fits.getdata(mskfile)
                except:
                    bdmskf = np.zeros((inst.nz, inst.ny, inst.nx), int)

                try: # Mips
                    brmskfile = re.sub(inst.bcdsuf, inst.brmsksuf, bcdfile)
                    brmskf = fits.getdata(brmskfile)
                except:
                    brmskf = -np.ones((inst.nz, inst.ny, inst.nx), int)

                # Obtain expid and dcenum
                index = pattern.search(bcd[i])
                expid = int(index.group(1))
                dcenum = int(index.group(2))

                if np.size(bdmskf) == 1:
                    bdmskf = np.zeros((inst.nz, inst.ny, inst.nx), int)
                if np.size(brmskf) == 1:
                    brmskf = -np.ones((inst.nz, inst.ny, inst.nx), int)

                # Find dither position
                pos = 0  # No dither position in stare data
                if 'DITHPOS' in bcdhead:
                    pos = bcdhead['DITHPOS'] - 1

                if inst.name == 'irs':
                    pos = expid % inst.npos
                elif inst.name == 'mips':
                    nod = expid % inst.nnod
                    pos = nod * inst.nscyc + mirind[bcdhead['CSM_PRED']]

                # Current beginning and end:
                be = nframes
                en = nframes + inst.nz

                # Store data
                data[be:en] = dataf.reshape((inst.nz, inst.ny, inst.nx))
                uncert[be:en] = uncf.reshape((inst.nz, inst.ny, inst.nx))
                bdmskd[be:en] = bdmskf.reshape((inst.nz, inst.ny, inst.nx))
                brmskd[be:en] = brmskf.reshape((inst.nz, inst.ny, inst.nx))
                # All the single numbers per frame that we care about
                fp.frmobs[be:en] = nframes + np.arange(inst.nz)
                fp.pos[be:en] = pos
                fp.aor[be:en] = aor
                fp.expid[be:en] = expid
                fp.dce[be:en] = dcenum
                fp.subarn[be:en] = np.arange(inst.nz)
                time[be:en] = (
                    bcdhead['MJD_OBS'] * u.d
                    + inst.framtime * (0.5 + np.arange(inst.nz)) * u.s
                )
                x_hcrs[be:en] = bcdhead['SPTZR_X']
                y_hcrs[be:en] = bcdhead['SPTZR_Y']
                z_hcrs[be:en] = bcdhead['SPTZR_Z']
                fp.pxscl1[be:en] = np.abs(bcdhead['PXSCAL1'])
                fp.pxscl2[be:en] = np.abs(bcdhead['PXSCAL2'])
                fp.zodi[be:en] = bcdhead['ZODY_EST']
                fp.ism [be:en] = bcdhead['ISM_EST']
                fp.cib [be:en] = bcdhead['CIB_EST']
                try:  # Telemetry:
                    telem.afpat2b [be:en] = bcdhead['AFPAT2B']
                    telem.afpat2e [be:en] = bcdhead['AFPAT2E']
                    telem.ashtempe[be:en] = bcdhead['ASHTEMPE'] + 273.0
                    telem.atctempe[be:en] = bcdhead['ATCTEMPE'] + 273.0
                    telem.acetempe[be:en] = bcdhead['ACETEMPE'] + 273.0
                    telem.apdtempe[be:en] = bcdhead['APDTEMPE'] + 273.0
                    telem.acatmp1e[be:en] = bcdhead['ACATMP1E']
                    telem.acatmp2e[be:en] = bcdhead['ACATMP2E']
                    telem.acatmp3e[be:en] = bcdhead['ACATMP3E']
                    telem.acatmp4e[be:en] = bcdhead['ACATMP4E']
                    telem.acatmp5e[be:en] = bcdhead['ACATMP5E']
                    telem.acatmp6e[be:en] = bcdhead['ACATMP6E']
                    telem.acatmp7e[be:en] = bcdhead['ACATMP7E']
                    telem.acatmp8e[be:en] = bcdhead['ACATMP8E']
                except:
                    pass
                try:  # MIPS telemetry:
                    telem.cmd_t_24[be:en] = bcdhead['CMD_T_24']
                    telem.ad24tmpa[be:en] = bcdhead['AD24TMPA']
                    telem.ad24tmpb[be:en] = bcdhead['AD24TMPB']
                    telem.acsmmtmp[be:en] = bcdhead['ACSMMTMP']
                    telem.aceboxtm[be:en] = bcdhead['ACEBOXTM'] + 273.0
                except:
                    pass

                # Store filename:
                fp.filename[be:en] = os.path.realpath(bcddir + bcd[i])

                # Store header:
                head[int(nframes//inst.nz)] = str(bcdhead)

                # Header position of the star:
                bcdhead["NAXIS"] = 2
                WCS = wcs.WCS(bcdhead, naxis=2)
                pix = WCS.wcs_world2pix([[self.ra.deg, self.dec.deg]], 0)
                fp.headx[be:en] = pix[0,0]
                fp.heady[be:en] = pix[0,1]
                nimpos[pos] += inst.nz
                nframes += inst.nz

                # Print to log and screen:
                log.msg(f"{aor:4d}{expid:7d}{dcenum:7d}{pos:7d}")

        # Observation mid time:
        fp.time = at.Time(time, format="mjd", scale="utc")
        # Barycentric location:
        fp.loc = coord.SkyCoord(
            x_hcrs*u.km, y_hcrs*u.km, z_hcrs*u.km,
            frame='hcrs', obstime=fp.time,
            representation_type='cartesian',
        ).transform_to(coord.ICRS)

        target = coord.SkyCoord(self.ra, self.dec).cartesian
        # Light-time travel from observatory (Spitzer) to barycenter:
        ltt = at.TimeDelta(
            np.sum(fp.loc.cartesian.xyz.T * target.xyz, axis=1)/ac.c)
        # Barycentric time:
        fp.btime = fp.time + ltt

        # Orbital phase:
        fp.phase = (fp.btime.tdb.jd - self.ephtime.tdb.jd) / self.period.value
        if np.ptp(fp.phase) < 1:  # Eclipse (phase[0]>0) or transit (phase[0]<0)
            fp.phase -= int(np.amax(fp.phase))
        else:  # Phase curve (phase[0] > 0)
            fp.phase -= int(np.amin(fp.phase))

        # Image index per position:
        for pos in range(inst.npos):
            fp.im[fp.pos==pos] = np.arange(nimpos[pos], dtype=np.double)

        # Set cycle number, visit within obs. set, frame within visit:
        if inst.name == "mips":
            fp.cycpos = np.trunc(fp.frmobs / (2*inst.ndcenum))
            fp.visobs = np.trunc(fp.frmobs / inst.ndcenum)
            fp.frmvis = np.trunc(fp.frmobs % inst.ndcenum)
        else:
            fp.cycpos = np.trunc(fp.frmobs // (inst.npos * inst.nmcyc*inst.nz))
            fp.visobs = np.trunc(fp.frmobs // (inst.nmcyc * inst.nz))
            fp.frmvis = fp.im % (inst.nmcyc * inst.nz)

        # Update event:
        self.data = data
        self.uncert = uncert
        self.bdmskd = bdmskd
        self.brmskd = brmskd
        self.head = head
        self.mask = ""
        self.fp = fp
        self.nimpos = nimpos


        # Now some final checks:
        # Source estimated position:
        self.srcest = np.zeros((2, inst.npos))
        for p in range(inst.npos):
            self.srcest[0,p] = np.mean(self.fp.heady[self.fp.pos==p])
            self.srcest[1,p] = np.mean(self.fp.headx[self.fp.pos==p])

        # Plot a reference image:
        for pos in range(inst.npos):
            i = np.where(self.fp.pos==pos)[0][0]
            plt.figure(101, (8,6))
            plt.clf()
            plt.imshow(
                self.data[i], interpolation='nearest', origin='lower',
                cmap=plt.cm.viridis)
            plt.plot(self.srcest[1,pos], self.srcest[0,pos],'k+', ms=12, mew=2)
            plt.xlim(-0.5, inst.nx-0.5)
            plt.ylim(-0.5, inst.ny-0.5)
            plt.colorbar()
            if inst.npos > 1:
                plt.title(f"{self.ID} reference image pos {pos}")
                plt.savefig(f"{self.ID}_sample-frame_pos{pos:02d}.png")
            else:
                plt.title(f"{self.ID} reference image")
                plt.savefig(f"{self.ID}_sample-frame.png")

        # Throw a warning if the source estimate position lies outside of
        # the image.
        out_of_bounds = (
            np.any(self.srcest[1,:] < 0)
            or np.any(self.srcest[1,:] > inst.nx)
            or np.any(self.srcest[0,:] < 0)
            or np.any(self.srcest[0,:] > inst.ny)
        )
        if out_of_bounds:
            log.warning("Source RA-DEC position lies out of bounds.")

        log.msg(
            f"\nSummary:\nTarget:     {self.planetname}\n"
            f"Event name: {self.ID}\n"
            f"Spitzer pipeline version: {inst.sscver}\n"
            f"AOR files: {inst.aorname}\nExposures per AOR: {inst.nexpid}\n"
            f"Number of target positions: {inst.npos}\n"
            f"Target guess position (pixels):\n {self.srcest}\n"
            f"Frames per position: {self.nimpos}\n"
            f"Read a total of {sum(self.nimpos):d} frames.\n"
        )

        print("Ancil Files:")
        if not os.path.isfile(inst.pmaskfile[0]):
            log.warning(
                f"Permanent mask file not found ('{inst.pmaskfile[0]}').")
        else:
            log.msg(f"Permanent mask file: '{inst.pmaskfile[0]}'")

        if not os.path.isfile(self.kurucz):
            log.warning(f"Kurucz file not found ('{self.kurucz}').")
        else:
            log.msg(f"Kurucz file: '{self.kurucz}'")

        if not os.path.isfile(self.filter):
            log.warning(f"Filter file not found ('{self.filter}').")
        else:
            log.msg(f"Filter file: '{self.filter}'")

        if not os.path.isfile(self.psf):
            log.warning(f"PSF file not found ('{self.psf}').")
        else:
            log.msg(f"PSF file: '{self.psf}'", indent=2)

        if self.inst.exptime is None:
            log.warning("Exposure time undefined.")
        if self.inst.gain is None:
            log.warning("Gain undefined.")

        io.save(self)


class Instrument:
    def __init__(self, inst):
        self.name = inst
        if self.name == "mips":
            self.chan = 6
            self.channel = "/ch1"
            self.prefix = "M"
        elif self.name == "irs":
            self.chan = 5
            self.channel = "/ch0"
            self.prefix = "S"
        elif self.name.startswith("irac"):
            self.chan = int(self.name[-1])
            self.channel = f"/ch{self.chan}"
            self.prefix = "I"
        else:
            print("Wrong instrument name.")

        wavelength = [3.6, 4.5, 5.8, 8.0, 16.0, 24.0]
        self.wavel = wavelength[self.chan-1] * u.Unit("micron")

        # Frequency calculated from wavelength
        self.freq = (ac.c/self.wavel).decompose()

        if self.name.startswith("irac"):
            self.bcdsuf   = '_bcd.fits'     # bcd image (basic calibrated data)
            self.buncsuf  = '_bunc.fits'    # bcd uncertainties
            # Note: At some point in time, SSC switched from dmask to imask,
            #  thus, this pipeline uses imask file, if not, then the dmask file.
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

        self.msaicsuf = '_msaic.fits'  # pbcd mosaic image
        self.msuncsuf = '_msunc.fits'  # pbcd mosaic uncertainties
        self.mscovsuf = '_mscov.fits'  # pbcd mosaic coverage (number of images)
        self.irsasuf  = '_irsa.tbl'    # list of 2MASS sources in the field
        self.pmasksuf = '_pmask.fits'  # pointing-refinement-corrected keywords

        # Permanent bad pixel mask:
        # irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/34
        # Sec 7.1 Darks, Flats, Missing data, and Bad Pixels:
        #   2** 6    Non-linear “rogue” pixels (IRS)
        #   2** 7    Dark current highly variable
        #   2** 8    Response to light highly variable
        #   2** 9    Pixel response to light is too high (fast saturation)
        #   2**10    Pixel dark current is too excessive
        #   2**14    Pixel response to light is too low (pixel is dead)
        self.pcrit = 65535

        # Individual bad pixel masks:
        if self.chan < 6:
            # in dmask (per-frame bad-pixel mask, IRACDH2.0, T4.2) added bit
            # 4 (decimal 16) since uncerts are high and flux is low in top
            # row, which has this flag
            # irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/56
            self.dcrit = (
                + 2** 4  #  saturation corrected in pipeline
                + 2** 5  #  muxbleed in ch 1, 2; bandwidth effect in ch 3, 4
                + 2** 8  #  crosstalk flag
                + 2** 9  #  radhit (single frame radhit detection)
                + 2**10  #  latent flag (not functional in IER observations)
                + 2**11  #  not flat-field corrected
                + 2**12  #  data not very linear
                + 2**13  #  saturated (not corrected in pipeline),
                         #   or predicted to be saturated in long HDR frames
                + 2**14  #  data bad and/or missing
            )
        else:
            # irsa.ipac.caltech.edu/data/SPITZER/docs/mips/mipsinstrumenthandbook/68/
            # FINDME: Check these:
            # 2** 9   Radhit detection (radhitmedfilt)
            # 2**10   Uncertainty Status
            # 2**11   Pixel masked in pmask - bad hardware state (satmask)
            # 2**12   Non-linearity correction could not be computed (slopecorr)
            # 2**13   Soft saturated (satmask)
            # 2**14   Data missing in downlink (cvti2r4)
            # 2**15   reserved: sign bit
            self.dcrit = 65024

        # Default ancilliary files:
        default_filters = [
            "irac1_filter.dat",
            "irac2_filter.dat",
            "irac3_filter.dat",
            "irac4_filter.dat",
            "irs-blue_filter.dat",
            "mips-24um_filter.dat",
        ]
        self.default_filter = default_filters[self.chan-1]

        default_psfs = [
            "IRAC.1.PRF.5X.070312.fits",
            "IRAC.2.PRF.5X.070312.fits",
            "IRAC.3.PRF.5X.070312.fits",
            "IRAC.4.PRF.5X.070312.fits",
            "IRS_BPUI_PSF.fits",
            "",
        ]
        self.default_psf = default_psfs[self.chan-1]


class Telemetry:
    """
    Spizter-specific temperature telemetry.
    """
    def __init__(self, nframes):
        self.afpat2b  = np.zeros(nframes)
        self.afpat2e  = np.zeros(nframes)
        self.ashtempe = np.zeros(nframes)
        self.atctempe = np.zeros(nframes)
        self.acetempe = np.zeros(nframes)
        self.apdtempe = np.zeros(nframes)
        self.acatmp1e = np.zeros(nframes)
        self.acatmp2e = np.zeros(nframes)
        self.acatmp3e = np.zeros(nframes)
        self.acatmp4e = np.zeros(nframes)
        self.acatmp5e = np.zeros(nframes)
        self.acatmp6e = np.zeros(nframes)
        self.acatmp7e = np.zeros(nframes)
        self.acatmp8e = np.zeros(nframes)
        self.cmd_t_24 = np.zeros(nframes)
        self.ad24tmpa = np.zeros(nframes)
        self.ad24tmpb = np.zeros(nframes)
        self.acsmmtmp = np.zeros(nframes)
        self.aceboxtm = np.zeros(nframes)


class FrameParameters:
    """
    class holder of the frame parameters.
    """
    def __init__(self, nframes):
        self.frmobs   = np.zeros(nframes, int)  # Frame number
        self.pos      = np.zeros(nframes, int)  # Position number
        self.aor      = np.zeros(nframes, int)  # AOR number
        self.expid    = np.zeros(nframes, int)  # Exposure ID
        self.dce      = np.zeros(nframes, int)  # Data Collection Event
        self.subarn   = np.zeros(nframes, int)  # Subarray frame number
        self.good     = np.zeros(nframes, bool) # frame tags in fp
        self.im       = np.zeros(nframes, int)  # Frame within position
        self.cycpos   = np.zeros(nframes, int)
        self.visobs   = np.zeros(nframes, int)
        self.frmvis   = np.zeros(nframes, int)
        self.zodi     = np.zeros(nframes)       # Zodiacal light estimate
        self.ism      = np.zeros(nframes)       # interstellar medium estimate
        self.cib      = np.zeros(nframes)       # Cosmic infrared background
        self.pxscl2   = np.zeros(nframes)
        self.pxscl1   = np.zeros(nframes)
        self.heady    = np.zeros(nframes)
        self.headx    = np.zeros(nframes)
        self.filename = np.zeros(nframes, dtype='S150')
        self.telemetry = Telemetry(nframes)
