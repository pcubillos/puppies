# Copyright (c) 2018 Patricio Cubillos and contributors.
# puppies is open-source software under the MIT license (see LICENSE).

import sys
import os
import configparser

from . import inst
from . import tools as pt


def init(cfile):
  """
  Parse variables from a configuration file into a dictionary.
  """
  # Parse variables from a configuration file into a dictionary.
  config = configparser.ConfigParser()
  config.optionxform=str
  config.read([cfile])

  if "PUPPIES" not in config.sections():
    pt.error("Invalid configuration file: '{:s}', no [PUPPIES] section.".
             format(cfile))
  # Extract inputs:
  args = dict(config.items("PUPPIES"))

  # Check args contains "telescope":
  if "telescope" not in args.keys():
    pt.error("Invalid configuration file: '{:s}', no 'telescope' parameter.".
             format(cfile))

  if args["telescope"] == "spitzer":
    pup = inst.Spitzer(args)
  elif args["telescope"] == "cheops":
    pup = None
  elif args["telescope"] == "jwst":
    pup = None

  return pup


def run(args):
  runmode = pt.parray(args["runmode"])
  nsteps = len(runmode)

  if runmode[0] == "load":
    pup = s.Pup(args)
  else:
    pup = ls.load(args["pickle"]) # Load object
    update(pup)

  for i in np.arange(nsteps):
    if runmode[i] == "badpix":
      pbp.badpix(pup)
    elif runmode[i] == "center":
      pass
    elif runmode[i] == "phot":
      pass
    elif runmode[i] == "model":
      pass
    elif runmode[i] == "mcmc":
      pass
