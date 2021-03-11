import pickle
import numpy as np

__all__ = [
    "save",
    "load",
    ]


def save(pup):
  """
  Save object into pickle file keeping specified variables into a
  separate npz file.
  """
  # List of variable to be saved into npz file:
  varnames = ["data", "uncert", "mask", "head", "bdmskd", "brmskd"]
  # Output npz file:
  savefile = "{:s}/{:s}.npz".format(pup.folder, pup.ID)
  # Info to be saved:
  info = dict()

  # Check for any of those variables:
  for varname in varnames:
      if hasattr(pup, varname):
          info[varname] = getattr(pup, varname)
          # Store the filename of saved arrays in varname + 'file':
          setattr(pup, f"{varname}file", savefile)
          # Remove from pup object:
          delattr(pup, varname)

  # Save data into npz file:
  if len(info) > 0:
    np.savez(savefile, **info)

  # Need to close this FILE object:
  pup.log.close()
  del pup.log

  # Pickle save:
  with open("{:s}/{:s}.p".format(pup.folder, pup.ID), "wb") as f:
      pickle.dump(pup, f)


def load(file, param=None):
  """
  Load a pickle file (if param is None) or load a specified parameter
  from a Numpy npz file.
  """
  if param is None:
    with open(file, "rb") as f:
      pup = pickle.load(f)
    return pup
  else:
    with np.load(file) as f:
      idx = f.files.index(param)
      val = f[param]
    return val
