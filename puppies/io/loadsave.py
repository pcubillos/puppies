import pickle
import numpy as np

__all__ = ["save", "load"]


def save(pup):
  """
st = '/Users/patriciocubillos/Dropbox/IWF/projects/2017_puppies/puppies/puppies/wa043b.nop'
with open(st, 'wb') as f:
  pickle.dump(pup, f)
  """
  if pup.data is not None:
    # Output file:
    savefile = "{:s}/{:s}.npz".format(pup.folder, pup.ID)
    # Info to be saved:
    info = dict()

    # List of variable to be saved into npz file:
    varnames = ["data", "uncert", "mask", "head", "bdmskd", "brmskd"]

    for i in np.arange(len(varnames)):
      var = getattr(pup, varnames[i])
      if not isinstance(var, str):
        info[varnames[i]] = var
        setattr(pup, varnames[i], savefile)

    #if not isinstance(pup.data, str):
    #  info["data"] = pup.data
    #  pup.data = savefile

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
