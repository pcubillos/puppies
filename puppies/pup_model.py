import os
import sys
import time
import numpy as np

from . import tools as pt
from . import io    as io
from . import image as im
from . import plots as pp

topdir = os.path.realpath(os.path.dirname(__file__) + "/../..")
sys.path.append(topdir + "/modules/MCcubed")
import MCcubed.utils as mu

"""
this module runs the light-curve modeling, either optimization or
Markov Chain Monte Carlo runs.

Main Routines
-------------
setup: Set up the event for MCMC simulation. Initialize the models
fit: Least-squares wrapper for ligtcurve fitting.
mcmc: Run the MCMC, produce plots, and save results. 

Auxiliary routines
------------------
evalmodel: Evaluate light-curve model with given set of parameters.
"""

"""
These are the rules:
- There are three types of runs: {SDNR, BIC, joint}
    BIC:   N model, 1 dataset, N mcmc
    SDNR:  1 model, N dataset, N mcmc
    joint: 1 model, N dataset, 1 mcmc
- joints are detected when using inter-dataset shared parameters
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
  laika.leastsq   = config[0]["leastsq"]
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
    pup.ndata = int(np.sum(good))
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

    # FINDME: Apply sigma-rejection mask

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

      # Setup models (in reverse order, because pixmap might modify mask):
      for k in np.arange(fit.nmodels[j])[::-1]:
        fit.models[j][k].setup(pup=pup, mask=fit.mask[j])

    fit.ndata.append(np.sum(fit.mask[j]))
    # Cumulative number of models for each pup:
    fit.iparams.append(iparams)

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
  - Re-do least-squares fitting with new uncertainties.
  - Save best-fitting parameters to initvals file.
  """
  if laika is None:
    if cfile is None:
      pt.error("Neither a config file nor a pup object was provided.")
    # Call setup
    setup(cfile)

  # Indices of non-fixed parameters:
  inonprior = np.where(stepsize > 0)[0] 

  # Run MC3's fit
  # Least-squares fit:
  if laika.leastsq:
    print("Calculating least-squares fit.")
    # calculate, now also fitting the parameters:
    output = modelfit(params, inonprior, stepsize, fit, full=True)

  # Evaluate model using current values:
  for j in range(npups):
    fit[j].fit0, fit[j].binipflux, fit[j].binipstd = \
                 evalmodel(params, fit[j], getbinflux=True, getbinstd=True)

    for k in np.arange(fit.nmodels[j]):
      if fit[j].models[k].type == 'pixmap':
        fit[j].binipflux = fit[j].binipflux.reshape(fit[j].gridshape)
        fit[j].binipstd  = fit[j].binipstd. reshape(fit[j].gridshape)

    # Reduced chi-square:
    fit[j].redchisq = np.sum(((fit[j].fit0 - fit[j].flux) / 
                      fit[j].sigma)**2.0) / (fit[j].nobj - fit[j].numfreepars)
    print("Reduced Chi-square: " + str(fit[j].redchisq), file=printout)

  # Since Spitzer over-estimates errors,
  # Modify sigma such that reduced chi-square = 1
  for j in np.arange(npups):
    # Uncertainty of data from raw light curve:
    fit[j].rawsigma      = fit[j].sigma
    # Scaled data uncertainty such reduced chi-square = 1.0
    fit[j].scaledsigma   = fit[j].sigma   * np.sqrt(fit[j].redchisq)

    if config[0].chi2flag:
      fit[j].sigma   = fit[j].scaledsigma  

  # New Least-squares fit using modified sigma values:
  if leastsq.leastsq:
    print("Re-calculating least-squares fit with new errors.")
    output = modelfit(params, inonprior, stepsize, fit, verbose=True)

  # Calculate current chi-square and store it in fit[0]:
  fit[0].chisq = 0.0
  for j in np.arange(npups):
    model = evalmodel(params, fit[j])
    fit[0].chisq += np.sum(((model - fit[j].flux)/fit[j].sigma)**2.0)
  # Store current fitting parameters in fit[0]:
  fit[0].fitparams = params

  for   j in np.arange(npups):
    for k in np.arange(fit.nmodels[j]):
      parlist[j][k][2][0] = params[fit[j].iparams[k]]
    # Update initial parameters:
    pe.write(fit[j].modelfile, parlist[j])
    ier = os.system("cp %s %s/." % (fit[j].modelfile, pup[j].folder))

  # Store results into a pickle file:
  pass


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


def evalmodel(params, fit,        getuc=False,     getipflux=False, 
              getbinflux=False,   getbinstd=False, getipfluxuc=False, 
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

  # Return statement:
  if (not getuc     and not getbinflux  and
      not getbinstd and not getipfluxuc and not getbinfluxuc):
    # Return the fit alone:
    return np.concatenate(lightcurve)

  # Else, return a list with the requested values:
  ret = [fit0]
  if getuc:
    ret.append(fituc0)
  if getbinflux:
    ret.append(binipflux)
  if getbinstd:
    ret.append(binstd)
  return ret

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


def residuals(freepars, params, inonprior, fit, nopriors=False):
  """
  Calculate the residual between the lightcurves and models.

  Parameters:
  -----------
  freepars: 1D ndarray
            Array of fitting light-curve model parameters.
  params: 1D ndarray 
            Array of light-curve model parameters.
  inonprior: 1D ndarray
            Array with the indices of freepars.
  fit: List of fits instances of an pup.
  nopriors: Boolean
            Do not add priors penalty to chi square in minimization.
  """
  # The output:
  residuals = []        # Residuals from the fit
  prior_residuals = []  # Residuals from the prior penalization

  # Number of pups:
  npups = len(fit)

  # Update the fitting parameters:
  pmin, pmax, stepsize  = [], [], []
  for j in np.arange(npups): 
    pmin     = np.concatenate((pmin,     fit[j].pmin),     0)
    pmax     = np.concatenate((pmax,     fit[j].pmax),     0)
    stepsize = np.concatenate((stepsize, fit[j].stepsize), 0)

  # Fitting parameters:
  params[inonprior] = freepars

  # Check min and max boundaries:
  params[np.where(params < pmin)] = pmin[np.where(params < pmin)]
  params[np.where(params > pmax)] = pmax[np.where(params > pmax)]

  # Update shared parameters:
  for i in np.arange(len(stepsize)):
    if stepsize[i] < 0:
      params[i] = params[-stepsize[i]-1]

  # Evaluate model for each pup:
  for j in np.arange(npups):
    model = evalmodel(params, fit[j])

    # Calculate pup's residuals and concatenate:
    ev_res = (model - fit[j].flux)/fit[j].sigma
    residuals = np.concatenate((residuals, ev_res))

    # Apply priors penalty if exists:
    if len(fit[j].ipriors) > 0:
      pbar = fit[j].priorvals[:,0]
      psigma = np.zeros(len(pbar))
      for i in np.arange(len(fit[j].ipriors)):
        if params[fit[j].ipriors[i]] < pbar[i]:
          psigma[i] = fit[j].priorvals[i,1]
        else:
          psigma[i] = fit[j].priorvals[i,2]
        #priorchisq += ((params[fit[j].ipriors[i]]-pbar[i])/psigma[i])**2.0
        prior_residuals.append((params[fit[j].ipriors[i]]-pbar[i])/psigma[i])

  # chisq = np.sum(residuals**2) + priorchisq
  # pseudoresiduals = np.sqrt(chisq/len(residuals)*np.ones(len(residuals)) )
  if nopriors:
    return residuals
  return np.concatenate((residuals, prior_residuals))


def modelfit(params, inonprior, stepsize, fit, verbose=False, full=False,
             retchisq=False, nopriors=False):
  """
  Least-squares fitting wrapper.

  Parameters:
  -----------
  params: 1D ndarray 
          Array of light-curve model parameters.
  inonprior: 1D ndarray
             Array with the indices of freepars.
  stepsize: 1D ndarray
            Array of fitting light-curve model parameters.
  fit: List of fits instances of an pup.
  verbose:  Boolean
            If True print least-square fitting message.
  full:     Boolean
            If True return full output in scipy's leastsq and print message.
  retchisq: Boolean
            Return the best-fitting chi-square value.
  nopriors: Boolean
            Do not add priors penalty to chi square in minimization.
  """
  fitting = op.leastsq(residuals, params[inonprior],
        args=(params, inonprior, fit, nopriors), factor=100, ftol=1e-16,
        xtol=1e-16, gtol=1e-16, diag=1./stepsize[inonprior], full_output=full)

  # Unpack least-squares fitting results:
  if full:
    output, cov_x, infodict, mesg, err = fitting
    print(mesg)
  else:
    output, err = fitting

  # Print least-squares flag message:
  if verbose:
    if (err >= 1) and (err <= 4):
      print("Fit converged without error.")
    else:
      print("WARNING: Error with least squares fit!")

  # Print out results:
  if full or verbose:
    print("Least squares fit best parameters:")
    print(output)

  # Return chi-square of the fit:
  if retchisq:
    fit_residuals = residuals(params[inonprior], params, inonprior, fit)
    chisq = np.sum(fit_residuals)**2.0
    return output, chisq

  return output


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
