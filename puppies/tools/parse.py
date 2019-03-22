# Copyright (c) 2018-2019 Patricio Cubillos and contributors.
# puppies is open-source software under the MIT license (see LICENSE).

__all__ = ['parse_model', 'parray']

import sys
import os
import configparser
import argparse

import numpy as np

from . import tools as pt


def parse_model(cfile, runmode="turtle"):
  """
  Parse a model config file.

  Parameters
  ----------
  cfile: String
     The input configuration file.
  runmode: String
     Running mode, select from [fit mcmc].

  Retuns
  ------
  A list of dictionaries, with one element per section.

  Some loose notes:
  - Use one section per dataset, labeled: MODEL, MODEL2, MODEL3, etc.
  - Datasets can be same visit but different reduction choice (center/photom)
  - Code should be able to run several fits to a same dataset.
    Multiple model fits are set by a newline in 'model' argument.
    (Multiple model fits allow for SDR, BIC comparison).
  - The code should be able to fit simultaneously different visits.
    (joint fits).
  - The first section must define all necessary variables.
  - Subsequent sections must set at least 'input' and 'modelfile'.  All other
    variables will be either 'inherited' from previous section or being
    redefined in the current section.
  """
  config = configparser.ConfigParser()
  config.optionxform=str
  config.read([cfile])

  # Get valid sections (each section is a different dataset):
  sections = config.sections()

  # Check if the config file is good at all:
  if not os.path.isfile(cfile):
    pt.error("Configuration file: '{:s}' not found.".format(cfile))
  if sections == []:
    pt.error(" The input configuration file '{:s}' has no valid sections.".
              format(cfile))

  # Check the sections are correctly set up:
  if sections[0] != "MODEL":
    pt.error("Invalid section name, the first section of the config file "
             "must be named 'MODEL'.")

  npups = len(sections)
  for i in range(1, npups):
    if "MODEL{:d}".format(i+1) not in sections:
      pt.error("Invalid section name, sections must be numbered "
               "in sequential order, e.g.: 'MODEL', 'MODEL2', 'MODEL3', ... ")

  # Check for required field in each pup:
  reqkeys = ["input", "model", "modelfile"]
  for key in reqkeys:
    for i in range(npups):
      if not config.has_option(sections[i], key):
        pt.error("Section '{:s}' does not contain the '{:s}' key.".
                  format(sections[i], key))

  # Output dictionary:
  pups = []
  pups.append(dict(config.items("MODEL")))
  pups[0] = defaults(pups[0])  # Data types and defaults
  for i in range(1, npups):
    # Copy previous entry:
    pups.append(dict(pups[i-1]))
    # Update values:
    pups[i].update(dict(config.items("MODEL{:d}".format(i+1))))
    pups[i] = defaults(pups[i])  # Data types and defaults

  return pups


def defaults(pupdict):
  """
  Set defaults and data types.
  """
  parser = argparse.ArgumentParser(description=__doc__, add_help=False,
                      formatter_class=argparse.RawDescriptionHelpFormatter)
  # Required option:
  add_arg(parser, "input",     str,    None)
  add_arg(parser, "model",     parray, None)
  add_arg(parser, "modelfile", parray, None)
  add_arg(parser, "output",    str,    None)
  add_arg(parser, "runmode",   str,    None)
  # Optimization options:
  add_arg(parser, "leastsq",   eval, False)
  add_arg(parser, "optimizer", str,  "lm")
  # MCMC options:
  add_arg(parser, "walk",     str,  None)
  add_arg(parser, "nsamples", eval, None)
  add_arg(parser, "burnin",   eval, None)
  add_arg(parser, "nchains",  int,  21)
  add_arg(parser, "nproc",    int,  None)
  add_arg(parser, "thinning", int,  None)
  add_arg(parser, "chiscale", eval, False)
  add_arg(parser, "grbreak",  int,  None)
  add_arg(parser, "grnmin",   int,  None)
  add_arg(parser, "plots",    eval, False)
  # Other fitting options:
  add_arg(parser, "joint",      eval,   False)
  add_arg(parser, "preclip",    parray, None)
  add_arg(parser, "postclip",   parray, None)
  add_arg(parser, "interclip",  parray, None)
  add_arg(parser, "sigrej",     parray, None)
  add_arg(parser, "priorvars",  parray, None)
  add_arg(parser, "priorvals",  parray, None)
  add_arg(parser, "tunits",     str,    None)
  add_arg(parser, "timeoffset", float,  0.0)
  add_arg(parser, "nbins",      int,    100)
  # BLISS options:
  add_arg(parser, "minpt", int,   4)
  add_arg(parser, "ystep", float, None)
  add_arg(parser, "xstep", float, None)

  # FINDME: Arguments that are not in this list will be kept as strings,
  # do I care?
  parser.set_defaults(**pupdict)
  args, unknown = parser.parse_known_args()
  # Put Namespace into a dict so we can extract their values:
  args = vars(args)

  # Check first pup contains all required fields (using argparser,
  # which will also set the default data types):
  if args['input'] is None:
    pt.error("Missing input.")

  # Make sure 'model' is a 2D nested list:
  if isinstance(args["model"][0], str):
    args["model"] = [args["model"]]

  return args


def add_arg(parser, dest, type, default=None):
  """
  A handy wrapper for argparse's add_argument() function.
  """
  parser.add_argument("--{:s}".format(dest), dest=dest, type=type,
                      default=default, action="store")


def parray(string, dtype=np.double):
  """
  Convert a string containin a list of white-space-separated and/or
  newline-separated values into a numpy array or list.

  Parameters
  ----------
  string: String

  Returns
  -------
  arr: ndarray or
  """
  if string == 'None':
    return None

  # Multiple lines:
  if string.find('\n') >= 0:
    string = string.split('\n')
    flatten = False
  else:
    string = [string]
    flatten = True

  arr = []
  for s in string:
    arr.append(s.split())

  if flatten:  # Input is 1D
    arr = arr[0]

  try:    # If they can be converted into doubles, do it:
    return np.array(arr, dtype)
  except: # Else, return a string array:
    return arr
