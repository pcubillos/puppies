# Copyright (c) 2021-2024 Patricio Cubillos
# puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

__all__ = [
    'newparams',
    'loadparams',
    'saveparams',
    ]

import os

import numpy as np

from .. import models as pm
from .  import tools  as pt


header = """\
# This file lists the models and their parameters.  The parameters are
# presented in a table headed by the name of each model. Column headers
# are unique to each model, but rows follow a standard format:
#   Row 1: Parameter value
#   Row 2: Lower bound
#   Row 3: Upper bound
#   Row 4: Step size
# NOTE1: To set one parameter equal to another, set its stepsize to the
# negative value of the index of the paramter you wish it to be equal to,
# starting from -1. For example: to set t12 = t34, set stepsize[3] = -5
# NOTE2: Zero stepsize results in a fixed value.\n
"""


def newparams(filename):
    """
    Create new modelparams text file with all currently-available
    models and their default values.

    Parameters
    ----------
    filename: String
       File where to save the model parameters.
    """
    # Make list of models:
    models = []
    for modelname in pm.__all__:
        models.append(eval("pm.{:s}()".format(modelname)))
    # Save to file:
    writeparams(filename, models)


def loadparams(filename, mnames=None):
    """
    Load up the info from a model-parameters file, extract models and
    values (params, bounds, and stepsizes).  If mnames is not None,
    return only the specified models.

    Parameters
    ----------
    filename: String
       Parameters file to load.
    mnames: List of strings
       List of model names to extract parameters.

    Returns
    -------
    models: List of model objects
       The requested models, initialized with filename values.
    """
    all_models = pm.__all__

    with open(filename, "r") as f:
        lines = f.readlines()
    # Skip header:
    i = 0
    while lines[i].startswith("#") or lines[i].strip() == "":
        i += 1

    # Update models' parameters with info from file:
    names, models = [], []
    while i < len(lines):
        modelname = lines[i].strip()
        if modelname not in all_models:
            print("Listed model ({:s}) is not in the list of available models.".
                format(modelname))
            i += 6
            continue
        m = eval("pm.{:s}()".format(modelname))
        try:
            m.params[:] = np.array(lines[i+2].split(), np.double)
            m.pmin  [:] = np.array(lines[i+3].split(), np.double)
            m.pmax  [:] = np.array(lines[i+4].split(), np.double)
            m.pstep [:] = np.array(lines[i+5].split(), np.double)
            names.append(modelname)
            models.append(m)
        except:
            pt.error(
                f"Incorrect number of parameters for model '{modelname}'. "
                f"This model should take {m.npars} parameters")
        i += 6

    # Select models if requested:
    if mnames is not None:
        missing = ~np.in1d(mnames, names)
        if np.any(missing):
            pt.error(
                "Some of the requested models ({:s}) are not in the "
                "list of available light curve models.".
                format(", ".join(np.array(mnames)[missing])))
        selected = []
        # Pick the selected models:
        for model in mnames:
            selected.append(models[names.index(model)])
        return selected
    return models


def saveparams(fit):
  """
  Update a modelparams file with current values from a puppies Fit()
  object.

  Parameters
  ----------
  fit: puppies Fit() object
  """
  for filename, fitmodels in zip(fit.modelfile, fit.models):
    # If file already exists, load for default values:
    defaults, dnames = [], []
    if os.path.isfile(filename):
      defaults = loadparams(filename)
    for dmodel in defaults:
      dnames.append(dmodel.name)

    for model in fitmodels:
      # Update with models from fit:
      if model.name in dnames:
        idx = dnames.index(model.name)
        defaults[idx] = model
      # This should never happen in principle:
      else:
        defaults.append(model)

    # Write to file:
    writeparams(filename, defaults)


def writeparams(filename, models):
  """
  Write models' info into filename.

  Parameters
  ----------
  filename: String
     Parameters file to write.
  models: List of puppies model() objects
     The models to write.
  """
  with open(filename, "w") as f:
    f.write(header)
    for model in models:
      # Model name:
      f.write("{:s}\n".format(model.name))
      # Parameter names, values, boundaries, and stepsizes:
      if model.npars == 0:  # Special case:
        f.write("CanIHazParz?\n")
        f.write("{:< 16.6e}\n""{:< 16.6e}\n""{:< 16.6e}\n""{:< 16.6e}\n"
                .format(0.0, 0.0, 0.0, 0.0))
      else:
        f.write("".join(["{:16s}".format(s) for s in model.pnames]) + "\n")
        f.write("".join(["{:< 16.6e}".format(p) for p in model.params]) + "\n")
        f.write("".join(["{:< 16.6e}".format(p) for p in model.pmin  ]) + "\n")
        f.write("".join(["{:< 16.6e}".format(p) for p in model.pmax  ]) + "\n")
        f.write("".join(["{:< 16.6e}".format(p) for p in model.pstep ]) + "\n")
