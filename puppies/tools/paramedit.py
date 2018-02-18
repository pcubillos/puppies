import os
import numpy as np

from .. import models as pm
from .  import tools  as pt


__all__ = ["loadparams"]


def init_comment():
    comment = """****COMMENT SPACE****
This file lists the models and their parameters.
The parameters are presented in a table headed
by the name of each model. Column headers are
unique to each model, but rows follow a
standard format:
  Row 1: Initial parameters
  Row 2: Lower bounds
  Row 3: Upper bounds
  Row 4: Step size
NOTE1: To set one parameter equal to another, set its stepsize to the
negative value of the location of the paramter you wish it to be equal to.
CAUTION: The first location is 1, NOT 0, since you can't have -0.
    eg. To set t12 = t34, set stepsize[3] = -5
NOTE2: Zero stepsize results in a fixed value.
****MODELS BELOW****
"""
    return comment


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
  # Skip comments:
  i = 0
  while lines[i].startswith("#"):
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
      pt.error("Incorrect number of parameters for model '{:s}'. This model"
               "should take {:d} parameters.".format(modelname, m.npars))
    i += 6

  # Select models if requested:
  if mnames is not None:
    missing = ~np.in1d(mnames, names)
    if np.any(missing):
      pt.error("Some of the requested models ({:s}) are not in the "
               "list of available light curve models.".
               format(", ".join(np.array(mnames)[missing])))
    selected = []
    # Pick the selected models:
    for i in np.arange(len(mnames)):
      selected.append(models[names.index(mnames[i])])
    return selected
  return models


def write(filename, models_used):
  """
  FINDME: Needs to be implemented.

  Parameters
  ----------
  filename: String
  models:
  """
  if not os.path.isfile(filename):
    print("'{:s}' does not exist.  Creating new file...".format(filename))
    models = read_parameters('params2.txt') # Read from default/backup file
    write_parameters(filename, models)
  else:
    models = read_parameters(filename)

  print('Updating the following models:')
  for i in models_used:
    k = 0
    print(i[0])
    for j in models:
      if i[0] == j[0]: #Test if model names are equal
        models[k] = i
      k+=1
  write_parameters(filename, models)
  return


def write_parameters(filename, models, verbose=False):
  """
  FINDME: Needs to be implemented.

  Parameters
  ----------
  filename: String
     Name of file where to write the parameters.
  models: List
  verbose: Bool
  """
  f = open(filename, "w")
  f.write(init_comment())
  for model in models:
    f.write(model[0] + "\n")  # Model name
    f.write(model[1] + "\n")  # Parameter names
    for j in np.arange(4):
      f.write("\t".join(["{:11.4e}".format(x) for x in model[2][j]]))
      f.write('\n')
  f.close()
  if verbose:
    print("Parameters have been written to '{:s}'.".format(filename))
  return

