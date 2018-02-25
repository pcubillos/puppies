import os
import sys
import time
import numpy as np

from . import tools as pt
from . import io    as io
from . import image as im
from . import plots as pp
from . import stats as ps

topdir = os.path.realpath(os.path.dirname(__file__) + "/../..")
sys.path.append(topdir + "/modules/MCcubed")
import MCcubed.fit   as mf
import MCcubed.utils as mu


__all__ = ["setup", "fit", "mcmc", "evalmodel"]


"""
this module runs the light-curve modeling, either optimization or
Markov Chain Monte Carlo runs.

There are three types of runs: {SDNR, BIC, joint}
    BIC:   N model, 1 dataset, N mcmc
    SDNR:  1 model, N dataset, N mcmc
    joint: 1 model, N dataset, 1 mcmc
"""


class Model():
  """
  Mother class that contains the pups (datasets) and fits.
  """
  def __init__(self):
    self.fit  = []
    self.pup  = []


class Fit():
  """
  Fit class that contains the fitting setup for a single fit run.
  It may apply to one or more (joint fit) pups.
  """
  def __init__(self, npups):
    self.npups     = npups
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
    self.flux = np.zeros(0, np.double)
    self.ferr = np.zeros(0, np.double)
    # MC3-related arrays:
    self.params = np.zeros(0, np.double)
    self.pmin   = np.zeros(0, np.double)
    self.pmax   = np.zeros(0, np.double)
    self.pstep  = np.zeros(0, np.double)
    self.prior  = np.zeros(0, np.double)
    self.prilo  = np.zeros(0, np.double)
    self.priup  = np.zeros(0, np.double)
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
  laika.nfits = np.zeros(laika.npups, int)
  for j in np.arange(laika.npups):
    laika.nfits[j] = len(config[j]['model'])

  # General variables (look only at the first entry):
  laika.folder    = config[0]["output"]
  laika.nsamples  = config[0]["nsamples"]
  laika.nchains   = config[0]["nchains"]
  laika.timeunits = config[0]["timeunits"]
  laika.sigrej    = config[0]["sigrej"]
  laika.leastsq   = config[0]["leastsq"]
  laika.optimizer = config[0]["optimizer"]
  laika.joint     = config[0]["joint"]

  # Total number of MCMC runs:
  if config[0]['joint']:
    laika.nruns = np.amax(laika.nfits)
  else:
    laika.nruns = np.sum(laika.nfits)

  # FINDME: if joint > 0 and nmodels[j] != nmodels[j-1]:
  #    pt.error("Number of models in each event does not match.")

  # Gather light-curve data:
  for j in np.arange(laika.npups):
    # Open input lightcurve:
    pup = io.load(config[0]["input"])
    # Plotting number of bins:
    pup.nbins = config[j]['nbins']

    # Read in frame-parameters data from masked raw light curve:
    good = pup.fp.good
    pup.ndata  = int(np.sum(good))
    pup.flux   = pup.fp.aplev[good]
    pup.ferr   = pup.fp.aperr[good]
    pup.y      = pup.fp.y    [good]
    pup.x      = pup.fp.x    [good]
    pup.phase  = pup.fp.phase[good]
    pup.date   = pup.fp.time [good]
    pup.pos    = pup.fp.pos  [good]
    pup.aor    = pup.fp.aor  [good]
    pup.frmvis = pup.fp.frmvis[good]

    pup.meanflux = np.mean(pup.flux)

    # Data clip-out ranges:
    clips = []
    if len(config[j]["preclip"]) == len(config[j]["model"]):
      preclip =  config[j]["preclip"][i]
    else:
      preclip = config[j]["preclip"][0]
    clips.append([0, int(preclip)])

    if len(config[j]["postclip"]) == len(config[j]["model"]):
      postclip = pup.ndata - config[j]["postclip"][i]
    else:
      postclip = pup.ndata - config[j]["postclip"][0]
    clips.append([int(postclip), pup.ndata])

    if config[j]['interclip'] is not None:
      clips = np.concatenate([clips, config[j]['interclip']], axis=0)
      for k in np.arange(len(config[j]['interclip'])):
        clips.append(config[j]['interclip'][k])

    # Good-data mask (True=good, False=bad):
    pup.mask = np.ones(pup.ndata, bool)
    # Mask out clips:
    for clip in clips:
      pup.mask[clip[0]:clip[1]] = False

    # FINDME: Intra-pixel mask:
    #fit.ipmask = np.ones(p.ndata, bool)
    #for m in np.arange(len(config[j]['ipclip'])):
    #  ipclip = config[j]['ipclip'][m]
    #  fit.ipmask[ipclip[0]:ipclip[1]] = False

    # Apply sigma-rejection mask:
    # FINDME: What if the LC varies significantly from start to end?
    if laika.sigrej is not None:
      pup.mask = ps.sigrej(pup.flux, laika.sigrej, mask=pup.mask)

    # Set orbital phase or days as unit of time:
    pup.timeoffset = config[j]['timeoffset']
    if laika.timeunits == 'phase':
      pup.time = pup.fp.phase[good]
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
      pup.ystep = pup.xrms
    pup.xstep = config[j]['xstep']
    if pup.xstep is None:
      pup.xstep = pup.xrms
    pup.minpt = config[j]['minpt']

    del pup.fp
    laika.pup.append(pup)

    # Set models for each pup:
    for i in np.arange(laika.nfits[j]):
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
    # Set MC3 params, bounds, steps, and priors:
    ntotal = 0
    for j in np.arange(fit.npups):
      # Get priors from config files:
      ipup = fit.ipup[j]
      pup = laika.pup[ipup]
      iparams = []
      print("{:s}:  {}".format(pup.ID, fit.mnames[j]))
      priorvars = config[ipup]['priorvars']
      priorvals = config[ipup]['priorvals']
      for k in np.arange(fit.nmodels[j]):
        npars = fit.models[j][k].npars
        # Params, boundaries, and steps from modelfile:
        fit.params = pt.cat(fit.params, fit.models[j][k].params)
        fit.pmin   = pt.cat(fit.pmin,   fit.models[j][k].pmin)
        fit.pmax   = pt.cat(fit.pmax,   fit.models[j][k].pmax)
        fit.pstep  = pt.cat(fit.pstep,  fit.models[j][k].pstep)
        # Priors from config file:
        prior = np.zeros(npars)
        prilo = np.zeros(npars)
        priup = np.zeros(npars)
        for m in np.arange(len(priorvars)):
          if priorvars[m] in fit.models[j][k].pnames:
            idx = fit.models[j][k].pnames.index(priorvars[m])
            prior[idx], prilo[idx], priup[idx] = priorvals[m]
        fit.prior = pt.cat(fit.prior, prior)
        fit.prilo = pt.cat(fit.prilo, prilo)
        fit.priup = pt.cat(fit.priup, priup)

        fit.nparams = pt.cat(fit.nparams, [ntotal+npars])
        iparams.append(np.arange(ntotal, ntotal+npars, dtype=int))
        ntotal += npars

      # Cumulative number of models for each pup:
      fit.iparams.append(iparams)
      # Setup models (in reverse order, because pixmap might modify mask):
      for k in np.arange(fit.nmodels[j])[::-1]:
        fit.models[j][k].setup(pup=pup, mask=fit.mask[j])
        # Number of free parameters for each pup:
        fit.nfree[j] += np.sum(fit.pstep[fit.iparams[j][k]]>0)

      # Number of non-masked datapoints:
      fit.ndata.append(np.sum(fit.mask[j]))
      # The fit's data as 1D concatenated arrays:
      fit.flux = pt.cat(fit.flux, laika.pup[ipup].flux[fit.mask[j]])
      fit.ferr = pt.cat(fit.ferr, laika.pup[ipup].ferr[fit.mask[j]])

    # Set data indices for each pup:
    fit.idata = np.zeros((fit.npups, len(fit.flux)), bool)
    ndata = 0
    for j in np.arange(fit.npups):
      fit.idata[j][ndata:ndata+fit.ndata[j]] = True
      ndata += fit.ndata[j]

    # Binned data:
    fit.binflux = [None] * fit.npups
    fit.binferr = [None] * fit.npups
    fit.bintime = [None] * fit.npups
    for j in np.arange(fit.npups):
      data   = pup.flux[fit.mask[j]]
      uncert = pup.ferr[fit.mask[j]]
      time   = pup.time[fit.mask[j]]
      binsize = int(fit.ndata[j]/pup.nbins)
      binflux, binferr, bintime = mu.binarray(data, uncert, time, binsize)
      fit.binflux[j], fit.binferr[j], fit.bintime[j] = binflux, binferr, bintime

      # Pre/post clipped binned data:
      #if fit.preclip > 0:
      #  fit[j].preclipflux  = fit[j].fluxuc [:fit[j].preclip]
      #  fit[j].preclipsigma = fit[j].sigmauc[:fit[j].preclip]
      #  fit[j].binprecstd, fit[j].binprecflux = bd.bindata(fit[j].nbins,
      #              std=[fit[j].preclipsigma], weighted=[fit[j].preclipflux], 
      #              binsize=binsize)

    # Print summary of fit:
    #print("Total observed points:      {:7d}".format(pup.ndata))
    #print("Raw light-curve points:     {:7d}".format(fit[j].nobj))
    #print("BLISS masked-out points:    {:7d}".
    #       format(np.sum(1-fit[j].minnumptsmask))
    #print("Fitted light-curve points:  {:7d}".format(fit[j].nobj))
  return laika


def fit(cfile=None, laika=None):
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
      pt.error("Neither a config file nor a pup object was provided.")
    # Call setup
    setup(cfile)

  lm = laika.optimizer == "lm"
  # Run MC3 least-squares fit:
  print("Calculating least-squares fit.")
  for fit in laika.fit:
    # Optimization:
    output = mf.modelfit(fit.params, evalmodel, fit.flux, fit.ferr,
        indparams=[fit], stepsize=fit.pstep, pmin=fit.pmin, pmax=fit.pmax,
        prior=fit.prior, priorlow=fit.prilo, priorup=fit.priup, lm=lm)

    # Evaluate model using current values:
    params = output[1]
    model = evalmodel(params, fit)
    for j in np.arange(fit.npups):
      idata = fit.idata[j]
      fit.bestfit[j] = model[idata]
      # Reduced chi-square for each pup:
      fit.rchisq[j] = np.sum(((fit.bestfit[j]-fit.flux[idata])
                             /fit.ferr[idata])**2) / (fit.ndata[j]-fit.nfree[j])
      print("Reduced chi-square: {:f}".format(fit.rchisq[j]))

      if laika.chiscale:
        # Scale uncertainties such reduced chi-square = 1.0:
        fit.ferr[fit.idata[i]] *= np.sqrt(fit.rchisq[i])

    # New Least-squares fit using modified sigma values:
    if laika.chiscale and fit.npups > 1:
      print("Re-calculating least-squares fit with new errors.")
      output = mf.modelfit(fit.params, evalmodel, fit.flux, fit.ferr,
          indparams=[fit], stepsize=fit.pstep, pmin=fit.pmin, pmax=fit.pmax,
          prior=fit.prior, priorlow=fit.prilo, priorup=fit.priup, lm=lm)

    # Store best-fitting parameters:
    fit.bestparams = output[1]
    model = evalmodel(fit.bestparams, fit)
    # Calculate chi-square:
    for j in np.arange(npups):
      idata = fit.idata[j]
      fit.bestfit[j] = model[idata]
      fit.chisq += np.sum(((fit.bestfit[j]-fit.flux[idata])/fit.ferr[idata])**2)

    # Save best-fitting paramters to file:
    for j in np.arange(npups):
    # FINDME: Implement pt.saveparams()
      pt.saveparams(fit) #fit[j].modelfile, parlist[j])

  # FINDME: Store results into a pickle file:
  pass
  # FINDME: What to return?


def mcmc(cfile, fit=None):
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
  # FINDME: Do I want an option to take self.fit ouput?
  # Call setup
  setup(cfile)
  # Run MC3's MCMC:
  pass
  # Store results into a pickle file:
  pass


def evalmodel(params, fit, getuc=False, getbinflux=False, getbinstd=False,
              getbinfluxuc=False, skip=[]):
  """
  Evaluate the light-curve model for a single pup.

  Parameters
  ----------
  params: 1D ndarray
          List of lightcurve parameters
  fit: A fit instance
  getuc: Boolean
         Evaluate and return lightcurve for unclipped data.
  getbinflux:    Boolean
                 Return binned ipflux map.
  getbinstd:     Boolean
                 Return binned ipflux standard deviation.
  getbinfluxcuc: Boolean
                 Return binned ipflux map for unclipped data.
  skip: List of strings
        List of names of models not to evaluate.

  Returns
  -------
  lightcurve: 1D float ndarray
     The fit light-curve model evaluated according to params.
     If there are more than one pup in fit, concatenate all models.
  """
  lightcurve = []
  for j in np.arange(fit.npups):
    fit0 = np.ones(fit.ndata[j])
    for k in np.arange(fit.nmodels[j]):
      if fit.models[j][k].type == 'pixmap':
        fit.models[j][k].model = fit0
      if fit.models[j][k].name not in skip:
        fit0 *= fit.models[j][k](params[fit.iparams[j][k]])
    lightcurve.append(fit0)

  return np.concatenate(lightcurve)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def runmcmc(fit, numit, walk, mode, grtest, printout, bound):
  """
  MCMC simulation wrapper.

  Code content
  ------------
  - Set up parameters, limits, etc.
  - Orthogonalize if requested
  - Run MCMC
  - Check for better fitting parameters.
  - Re-run least-squares if necessary.
  - Calculate and print acceptance rate.

  Parameters
  ----------
  pup: List of pup instances
  fit: List of fit instances of an pup.
    numit: Scalar
           Number of MCMC iterations.
  walk: String
        Random walk for the Markov chain: 
        'demc': differential evolution MC.
        'mrw': Metropolis random walk.
  mode: String
           MCMC mode: ['burn' | 'continue' | 'final']
  grtest: Boolean
          Do Gelman and Rubin convergence test. 
  printout: File object for directing print statements
  bound: Boolean
         Use bounded-eclipse constrain for parameters (start after the 
         first frame, end before the last frame).
  """
  npups = len(fit)

  # Recover the last burn-in state to use as starting point:
  if mode == 'burn':
    ninitial = 1   # Number of initial params. sets to start MCMC
  else:
    ninitial = laika.nchains

  fitpars = np.zeros((ninitial, 0))
  pmin, pmax, stepsize = [], [], []
  ntotal = 0 # total number of parameters:
  for j in np.arange(npups):
    #fit[j].chainend  = np.atleast_2d(params[fit[j].indparams])
    fitpars   = np.hstack((fitpars, fit[j].chainend))
    pmin      = np.concatenate((pmin,     fit[j].pmin),      0)
    pmax      = np.concatenate((pmax,     fit[j].pmax),      0)
    stepsize  = np.concatenate((stepsize, fit[j].stepsize),  0)
    ntotal += fit[j].nump

  inonprior = np.where(stepsize > 0)[0]

  # Fix IP mapping to bestmip if requested (for final run):
  if mode == 'final':
    for j in np.arange(npups):
      if hasattr(config[j], 'isfixipmap') and config[j].isfixipmap:
        for k in np.arange(fit.nmodels[j]):
          if fit[j].models[k].type == 'pixmap':
            fit[j].funcx.append([fit[j].bestmip, fit[j].binipflux,
                                 np.zeros(len(fit[j].wbfipmask))])
            fit[j].funcs[k] = mc.fixipmapping

  # Run MCMC:
  allparams, numaccept, bestp, bestchisq = mcmc.mcmc(fitpars, pmin, pmax,
                              stepsize, numit, fit,
                              laika.nchains, walk=walk, 
                              grtest=grtest, bound=bound)
  numaccept = np.sum(numaccept)

  if mode == 'final':
    for j in range(npups):
      # Record trace of decorrelated parameters, if any, otherwise
      # same as fit[j].allparams
      fit[j].allparams = np.copy(allparams[:,fit[j].indparams])
      fit[j].bestp     = np.copy(bestp    [  fit[j].indparams])
    # Reshape allparams joining chains for final results:
    allp = np.zeros((ntotal, numit*laika.nchains))
    for par in np.arange(ntotal):
      allp[par] = allparams[:,par,:].flatten()
    del(allparams)
    allparams = allp

  for j in np.arange(npups):
    # Save chains end point:
    fit[j].chainend = allparams[:,fit[j].indparams,-1]

  return allparams
