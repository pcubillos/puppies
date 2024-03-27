# Copyright (c) 2021-2024 Patricio Cubillos
# puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

__all__ = [
    'setup',
    'lightcurve_fit',
    'mcmc',
    'evalmodel',
]

import os

import numpy as np
import mc3

from .. import tools as pt
from .. import io as io
from .. import plots as pp
from .. import stats as ps


class Model():
    """Mother class that contains the pups (datasets) and fits."""
    def __init__(self):
        self.fit = []
        self.pup = []

    def summary(self, units="percent"):
        if units == "percent":
            fmt = "6.4f"
            fac = 1e2
            units = f"({units})"
        elif units == "ppm":
            fmt = "5.0f"
            fac = 1e6
            units = f"({units})"
        elif units == "":
            fmt = "8.6f"
            fac = 1.0

        depths = ["rprs", "depth", "idepth", "edepth"]
        k = 0
        print(f"\n{'Target':6s}  {'Observation':20s}  Reduction", end="")
        for i in range(self.npups):
            pup = self.pup[i]
            jmin = np.argmin(self.bic[i])
            #obs = "".join([pup.telescope, pup.inst.name, pup.visit])
            obs = "/".join(["Spitzer", pup.inst.name])
            red = pup.folder[2+len(pup.root)+len(pup.ID):]
            print(f"\n{pup.ID:6s}  {obs:20s}  {red}")

            if self.fit[0].errorlow is not None:
                slen = 2*int(fmt[0]) + 9
                print(f"\n     SDR     dBIC          {units:{slen}s}S/N  Fit")
            else:
                print(f"\n     SDR     dBIC          {units:9s}  Fit")
            for j in range(self.nfits[i]):
                fit = self.fit[k]
                dbic = self.bic[i][j] - self.bic[i][jmin]
                stats  = ["{:8.4f}{:9.2f}".format(self.sdr[i][j], dbic), ""]
                models = ["+".join(fit.mnames[0]), ""]
                s = []
                # Depths with error bars:
                if fit.errorlow is not None:
                    uncert = 0.5 * (fit.errorhigh-fit.errorlow)
                    for n in range(fit.nparams[-1]):
                        if fit.pnames[n] in depths:
                            s += ["{:>6s}: {:{}} +/- {:{}}  {:5.1f}".format(
                                fit.pnames[n],
                                fac*fit.bestparams[n], fmt,
                                fac*uncert[n], fmt,
                                (fit.bestparams/uncert)[n])]
                # Depths without error bars:
                else:
                    for n in range(fit.nparams[-1]):
                        if fit.pnames[n] in depths:
                            s += ["{:>6s}: {:9{}}".format(
                                fit.pnames[n],
                                fac*fit.bestparams[n],
                                fmt[1:])]
                for n in range(len(s)):
                    print(f"{stats[n!=0]:17s}  {s[n]}  {models[n!=0]}")
                k += 1


class Fit():
    """
    Class that contains the fitting setup for a single fit run.
    It may apply to one or more pups (joint fit).
    """
    def __init__(self, npups):
        self.npups = npups
        # Each of the following varables will have length npups:
        self.ipup      = []
        self.mnames    = []
        self.nmodels   = []
        self.modelfile = []
        self.models    = []
        self.mask      = []
        # Fit outputs:
        self.bestfit = [None] * npups
        self.chisq  = np.zeros(npups)
        self.rchisq = np.zeros(npups)
        # Data (filled as a 1D array with pup-concatenated non-masked data):
        self.flux  = np.array([], np.double)
        self.ferr  = np.array([], np.double)
        self.cferr = np.array([], np.double)
        # MC3-related arrays:
        self.params = np.array([], np.double)
        self.pmin   = np.array([], np.double)
        self.pmax   = np.array([], np.double)
        self.pstep  = np.array([], np.double)
        self.pnames = np.array([], np.double)
        self.prior  = np.array([], np.double)
        self.prilo  = np.array([], np.double)
        self.priup  = np.array([], np.double)
        self.parstd    = None
        self.errorlow  = None
        self.errorhigh = None
        # Parameters indexing:
        self.nparams = [0]
        self.iparams = []
        self.ndata   = []  # Not to confuse with pup.ndata
        self.nfree   = np.zeros(npups, int)


def setup(cfile, mode='turtle'):
    """
    Set up the event for MCMC simulation.  Initialize the models and
    performs a least-squares fit.

    Parameters
    ----------
    cfile: String
       A PUPPIES model config file.

    Content
    -------
    - Initialize laika, pup, and fit instances.
    - Read-in pups and config file.
    - Setup time units.
    - Mask data.
    - Initialize models.
    - Set up independent parameters for light-curve models.
    - Create binned time-data.
    """
    # Parse configuration file:
    config = pt.parse_model(cfile)

    # laika is the mother object that contains all the pups (and fits):
    laika = Model()
    # Number of datasets (SDNR comparison or shared fit):
    laika.npups = len(config)
    # Number of model fits per dataset (for model BIC comparison):
    laika.nfits = np.array([len(cfg['model']) for cfg in config])

    # General variables (look only at the first entry):
    laika.folder    = config[0]["output"]
    laika.walk      = config[0]["walk"]
    laika.nsamples  = config[0]["nsamples"]
    laika.burnin    = config[0]["burnin"]
    laika.nchains   = config[0]["nchains"]
    laika.ncpu      = config[0]["ncpu"]
    laika.thinning  = config[0]["thinning"]
    laika.timeunits = config[0]["timeunits"]
    laika.sigrej    = config[0]["sigrej"]
    laika.grbreak   = config[0]["grbreak"]
    laika.grnmin    = config[0]["grnmin"]
    laika.leastsq   = config[0]["leastsq"]
    laika.optimizer = config[0]["optimizer"]
    laika.chiscale  = config[0]["chiscale"]
    laika.joint     = config[0]["joint"]
    laika.plots     = config[0]["plots"]
    laika.resume    = False

    # Total number of MCMC runs:
    if config[0]['joint']:
        laika.nruns = np.amax(laika.nfits)
    else:
        laika.nruns = np.sum(laika.nfits)

    # FINDME: if joint > 0 and nmodels[j] != nmodels[j-1]:
    #    pt.error("Number of models in each event does not match.")

    # Gather light-curve data:
    for j in range(laika.npups):
        # Open input lightcurve:
        pup = io.load(config[j]["input"])
        # Plotting number of bins:
        pup.nbins = config[j]['nbins']

        # Read in frame-parameters data from masked raw light curve:
        good = pup.fp.good
        pup.ndata  = int(np.sum(good))
        pup.flux   = pup.fp.aplev[good]
        pup.ferr   = pup.fp.aperr[good]
        pup.y      = pup.fp.y[good]
        pup.x      = pup.fp.x[good]
        pup.phase  = pup.fp.phase[good]
        pup.date   = pup.fp.time[good]
        pup.pos    = pup.fp.pos[good]
        pup.aor    = pup.fp.aor[good]
        pup.frmvis = pup.fp.frmvis[good]

        pup.meanflux = np.mean(pup.flux)

        # Data clip-out ranges:
        clips = []
        preclip = config[j]["preclip"][0]
        clips.append([0, int(preclip)])

        postclip = pup.ndata - config[j]["postclip"][0]
        clips.append([int(postclip), pup.ndata])

        if config[j]['interclip'] is not None:
            clips = np.vstack([clips, config[j]['interclip']])
        clips = np.array(clips, int)

        # Who is a good mask? (True=good, False=bad):
        pup.mask = np.ones(pup.ndata, bool)
        # Mask out clips:
        for clip in clips:
            pup.mask[clip[0]:clip[1]] = False

        # FINDME: Intra-pixel mask:
        #fit.ipmask = np.ones(p.ndata, bool)
        #for m in range(len(config[j]['ipclip'])):
        #  ipclip = config[j]['ipclip'][m]
        #  fit.ipmask[ipclip[0]:ipclip[1]] = False

        # Apply sigma-rejection mask:
        # FINDME: What if the LC varies significantly from start to end?
        if laika.sigrej is not None:
            pup.mask = ps.sigrej(pup.flux, laika.sigrej, mask=pup.mask)
            pup.mask = ps.sigrej(pup.ferr, laika.sigrej, mask=pup.mask)

        # Set orbital phase or days as unit of time:
        pup.timeoffset = config[j]['timeoffset']
        if laika.timeunits == 'phase':
            pup.time = pup.fp.phase[good] - pup.timeoffset
            pup.xlabel = 'Orbital phase'
            pup.timeoffset = 0.0  # No offset for phase units
        elif laika.timeunits == 'jd':
            pup.time = pup.fp.time[good].jd  - pup.timeoffset
            pup.xlabel = 'BJD_UTC'
        elif laika.timeunits == 'mjd':
            pup.time = pup.fp.time[good].mjd - pup.timeoffset
            pup.xlabel = 'BMJD_UTC'

        if pup.timeoffset != 0.0:
            pup.xlabel += ' - {:.1f} days'.format(pup.timeoffset)

        # BLISS setup variables:
        pup.ystep = config[j]['ystep']
        if pup.ystep is None:
            pup.ystep = pup.yrms
        pup.xstep = config[j]['xstep']
        if pup.xstep is None:
            pup.xstep = pup.xrms
        pup.minpt = config[j]['minpt']

        del(pup.fp)
        laika.pup.append(pup)

        # Set models for each pup:
        for i in range(laika.nfits[j]):
            # Initialize fit class:
            if laika.joint and j==0:
                fit = Fit(laika.npups)
                laika.fit.append(fit)
            elif not laika.joint:
                fit = Fit(1)
                laika.fit.append(fit)
            else: # joint  and  j!=0:
                fit = laika.fit[i]

            # Dataset index:
            fit.ipup.append(j)
            fit.mask.append(np.copy(laika.pup[j].mask))
            # List of lightcurve model names for this fit/dataset:
            mnames = config[j]['model'][i]
            fit.mnames.append(mnames)
            fit.nmodels.append(len(mnames))

            # Load models' parameters:
            if len(config[j]["modelfile"]) == len(config[j]["model"]):
                modelfile = config[j]["modelfile"][i]
            else:
                modelfile = config[j]["modelfile"][0]
            fit.modelfile.append(modelfile)
            fit.models.append(pt.loadparams(modelfile, mnames))


    # FINDME: Move pixmap to last position in list of models?
    #if np.where(funct == 'pixmap') != (nmodels[j] - 1):
    #  pass

    # Now, focus on the fits:
    for fit in laika.fit:
        print("\nModel fit:")
        # Set params, bounds, steps, and priors:
        ntotal = 0
        for j in range(fit.npups):
            # Get priors from config files:
            ipup = fit.ipup[j]
            pup = laika.pup[ipup]
            iparams = []
            priorvars = config[ipup]['priorvars']
            priorvals = config[ipup]['priorvals']
            for k in range(fit.nmodels[j]):
                npars = fit.models[j][k].npars
                # Params, boundaries, and steps from modelfile:
                fit.params = pt.cat((fit.params, fit.models[j][k].params))
                fit.pmin   = pt.cat((fit.pmin,   fit.models[j][k].pmin))
                fit.pmax   = pt.cat((fit.pmax,   fit.models[j][k].pmax))
                fit.pstep  = pt.cat((fit.pstep,  fit.models[j][k].pstep))
                fit.pnames = pt.cat((fit.pnames, fit.models[j][k].pnames))
                # Priors from config file:
                prior = np.zeros(npars)
                prilo = np.zeros(npars)
                priup = np.zeros(npars)
                if priorvars is not None:
                    for m in range(len(priorvars)):
                        if priorvars[m] in fit.models[j][k].pnames:
                            idx = fit.models[j][k].pnames.index(priorvars[m])
                            prior[idx], prilo[idx], priup[idx] = priorvals[m]
                fit.prior = pt.cat((fit.prior, prior))
                fit.prilo = pt.cat((fit.prilo, prilo))
                fit.priup = pt.cat((fit.priup, priup))

                fit.nparams = pt.cat((fit.nparams, [ntotal+npars]))
                iparams.append(np.arange(ntotal, ntotal+npars, dtype=int))
                ntotal += npars

            # Cumulative number of models for each pup:
            fit.iparams.append(iparams)
            # Setup models (in reverse order, because pixmap might modify mask):
            for k in range(fit.nmodels[j])[::-1]:
                fit.models[j][k].setup(obj=pup, mask=fit.mask[j])
                # Number of free parameters for each pup:
                fit.nfree[j] += np.sum(fit.pstep[fit.iparams[j][k]]>0)

            # Number of non-masked datapoints:
            fit.ndata.append(np.sum(fit.mask[j]))
            # The fit's data as 1D concatenated arrays:
            fit.flux = pt.cat((fit.flux, laika.pup[ipup].flux[fit.mask[j]]))
            fit.ferr = pt.cat((fit.ferr, laika.pup[ipup].ferr[fit.mask[j]]))

        # Set data indices for each pup:
        fit.idata = np.zeros((fit.npups, len(fit.flux)), bool)
        ndata = 0
        for j in range(fit.npups):
            fit.idata[j][ndata:ndata+fit.ndata[j]] = True
            ndata += fit.ndata[j]

        # Binned data:
        fit.binflux = [None] * fit.npups
        fit.binferr = [None] * fit.npups
        fit.bintime = [None] * fit.npups
        for j in range(fit.npups):
            ipup = fit.ipup[j]
            pup = laika.pup[ipup]
            data = pup.flux[fit.mask[j]]
            uncert = pup.ferr[fit.mask[j]]
            time = pup.time[fit.mask[j]]
            binsize = int(fit.ndata[j]/pup.nbins)
            fit.bintime[j] = mc3.stats.bin_array(time, binsize)
            fit.binflux[j], fit.binferr[j] = mc3.stats.bin_array(
                data, binsize, uncert)

        # Print summary of fit:
        #print("Total observed points:      {:7d}".format(pup.ndata))
        #print("Raw light-curve points:     {:7d}".format(fit[j].nobj))
        #print("BLISS masked-out points:    {:7d}".
        #       format(np.sum(1-fit[j].minnumptsmask))
        #print("Fitted light-curve points:  {:7d}".format(fit[j].nobj))
    return laika


def lightcurve_fit(cfile=None, laika=None, summary=False):
    """
    Run the optimization for the lightcurves and models specified in the
    input configuration file.

    - Do Least-square fitting with initial parameters.
    - Scale data uncertainties to get reduced chisq == 1.0
    - FINDME: Do another sigrej on residuals?
    - Re-do least-squares fitting with new uncertainties.
    - Save best-fitting parameters to initvals file.

    """
    if laika is None:
        if cfile is None:
            pt.error("Neither a config file nor a Model object was provided.")
        # Call setup
        laika = setup(cfile)

    # Some stats:
    sdr = [[] for n in range(laika.npups)]
    bic = [[] for n in range(laika.npups)]

    # Run MC3 least-squares fit:
    print("Calculating least-squares fit.")
    for fit in laika.fit:
        output = mc3.fit(
            fit.flux, fit.ferr, evalmodel, fit.params, indparams=[fit],
            pstep=fit.pstep, pmin=fit.pmin, pmax=fit.pmax,
            prior=fit.prior, priorlow=fit.prilo, priorup=fit.priup,
            leastsq=laika.optimizer,
        )

        # Evaluate model using current values:
        fit.bestparams = output['bestp'] # bestp
        model = evalmodel(fit.bestparams, fit)
        for j in range(fit.npups):
            idata = fit.idata[j]
            fit.bestfit[j] = model[idata]
            # Reduced chi-square for each pup:
            fit.rchisq[j] = np.sum(
                (fit.bestfit[j]-fit.flux[idata])**2/fit.ferr[idata]**2) \
                / (fit.ndata[j]-fit.nfree[j])
            print(f"Reduced chi-square: {fit.rchisq[j]:.6f}")
            # Standard deviation of the residuals:
            sdr[fit.ipup[j]].append(np.std(fit.bestfit[j]-fit.flux[idata]))

    for fit in laika.fit:
        # Chi-square corrected flux error:
        fit.cferr = np.copy(fit.ferr)
        if laika.chiscale:
            for j in range(fit.npups):
                imin = np.argmin(sdr[fit.ipup[j]])
                # Scale uncertainties such reduced chi-square = 1.0
                #  (scale relative to fit with lowest SDR):
                fit.cferr[fit.idata[j]] = \
                    fit.ferr[fit.idata[j]] * np.sqrt(laika.fit[imin].rchisq[j])

            # New Least-squares fit using modified uncertainties:
            if laika.joint:
                print("Re-calculating least-squares fit with new errors.")
                output = mc3.fit(
                    fit.flux, fit.cferr, evalmodel, fit.params, indparams=[fit],
                    pstep=fit.pstep, pmin=fit.pmin, pmax=fit.pmax,
                    prior=fit.prior, priorlow=fit.prilo, priorup=fit.priup,
                    leastsq=laika.optimizer,
                )
                fit.bestparams = output['bestp']

        # Store best-fitting parameters:
        model = evalmodel(fit.bestparams, fit, update=True)
        # Calculate chi-square:
        for j in range(fit.npups):
            idata = fit.idata[j]
            fit.bestfit[j] = model[idata]
            chisq = np.sum(
                ((fit.bestfit[j]-fit.flux[idata])/fit.cferr[idata])**2)
            fit.chisq += chisq
            bic[fit.ipup[j]].append(chisq + fit.nfree[j]*np.log(fit.ndata[j]))

        # Save best-fitting paramters to file:
        for j in range(fit.npups):
            pt.saveparams(fit)

    # FINDME: Store results into a pickle file:
    pass

    laika.sdr = sdr
    laika.bic = bic
    if summary:
        laika.summary("percent")

    return laika


def mcmc(cfile=None, laika=None):
    """
    Run MCMC for the lightcurves and models specified in the input
    configuration file.

    Parameters
    ----------
    cfile: String
       A PUPPIES model config file.
    fit: String
       A PUPPIES model-fit pickle file (if not None, overrides cfile).
    """
    if laika is None:
        if cfile is None:
          pt.error("Neither a config file nor a Model object was provided.")
        # Call setup
        laika = setup(cfile)

    # Run optimization if requested:
    if laika.leastsq:
        laika = lightcurve_fit(laika=laika, summary=False)

    if not os.path.exists(laika.folder):
        os.mkdir(laika.folder)

    # Run MCMC:
    for k in range(len(laika.fit)):
        fit = laika.fit[k]
        if laika.joint:
            savefile = laika.folder
        else:
            pup = laika.pup[fit.ipup[0]]
            savefile = f"{laika.folder}/MCMC_{pup.centering}{pup.photap:.2f}_fit{k:02d}"
        laika.grbreak = 0.0

        posterior = mc3.sample(
            data=fit.flux, uncert=fit.cferr,
            func=evalmodel, params=fit.params, indparams=[fit],
            pmin=fit.pmin, pmax=fit.pmax, pstep=fit.pstep,
            prior=fit.prior, priorlow=fit.prilo, priorup=fit.priup,
            sampler=laika.walk, nsamples=laika.nsamples, burnin=laika.burnin,
            nchains=laika.nchains, ncpu=laika.ncpu, thinning=laika.thinning,
            grtest=True, grbreak=laika.grbreak, grnmin=0.5,
            hsize=10, kickoff='normal',
            plots=laika.plots,
            pnames=fit.pnames, resume=laika.resume, rms=True,
            log=f"{savefile}.log",
            savefile=f"{savefile}.npz",
        )
        # Need to eval with bestp to ensure that ipflux is correct
        bestfit = evalmodel(fit.bestparams, fit)
        # Store results into object:
        fit.errorlow = posterior['CRlo']  # Low credible-interval boundary
        fit.errorhigh = posterior['CRhi']  # High credible-interval boundary
        fit.parstd = posterior['stdp']  # Posterior standard deviation

        # Light-curve plot:
        if laika.plots:
            savefile_lc = f'{savefile}_lc_ramp.png'
            pp.lightcurve(fit, laika.pup, savefile_lc, systematics='ramp')
            savefile_lc = f'{savefile}_lc_corrected.png'
            pp.lightcurve(fit, laika.pup, savefile_lc, systematics='corrected')

    # Print MCMC results:
    laika.summary(units="percent")
    # TBD: Store results into a pickle file

    return laika


def evalmodel(params, fit, skip=[], update=False):
    """
    Evaluate the light-curve model for a single pup.

    Parameters
    ----------
    params: 1D ndarray
       List of lightcurve parameters.
    fit: A puppies Fit() instance
       Object containing the models to use.
    skip: List of strings
       List of names of models not to evaluate.
    update: Bool
       If True, set model.params from the params array.

    Returns
    -------
    lightcurve: 1D float ndarray
       The fit light-curve model evaluated according to params.
       If there is more than one pup in fit, concatenate all models.
    """
    lightcurve = []
    for j in range(fit.npups):
        fit0 = np.ones(fit.ndata[j])
        for k in range(fit.nmodels[j]):
            if fit.models[j][k].type == 'pixmap':
                fit.models[j][k].model = fit0
            if fit.models[j][k].name not in skip:
                fit0 *= fit.models[j][k](params[fit.iparams[j][k]])
            if update:
                fit.models[j][k].params = params[fit.iparams[j][k]]
        lightcurve.append(fit0)

    return np.concatenate(lightcurve)
