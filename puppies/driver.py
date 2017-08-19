import configparser
import sys, os

#import instrument as inst

from . import spitzer as s
from . import tools as pt


def init(cfile):
  """
  Parse variables from a configuration file into a dictionary.

import configparser, imp
import spitzer as s
import loadsave as ls
import pup_badpix as pbp

pup = ls.load("/Users/patriciocubillos/Dropbox/IWF/projects/2017_puppies/run00/wa043b/wa043b.p")
pbp.badpix(pup)

cfg = driver.parse("/Users/patriciocubillos/Dropbox/IWF/projects/2017_puppies/run00/wa043b/badpix/")
pup = ls.load("/Users/patriciocubillos/Dropbox/IWF/projects/2017_puppies/run00/wa043b/badpix/wa043b.p")
  """
  args = parse(cfile)

  # FINDME: check args contains "telescope"
  if args["telescope"] == "spitzer":
    pup = s.Pup(args)
  elif args["telescope"] == "cheops":
    pup = None
  elif args["telescope"] == "jwst":
    pup = None

  return pup


def parse(cfile):
  """
  Parse variables from a configuration file into a dictionary.

cfile='/Users/patriciocubillos/Dropbox/IWF/projects/2017_puppies/run00/pup.cfg'
args = parse(cfile)
  """
  config = configparser.ConfigParser()
  config.optionxform=str
  config.read([cfile])

  if "puppy" not in config.sections():
    pt.error("Invalid configuration file: '{:s}', no [puppy] section.".
             format(args.cfile))

  # Extract inputs:
  args = dict(config.items("puppy"))
  return args


def run(args):
  runmode = pt.parray(args["runmode"])
  nsteps = len(runmode)

  if runmode[0] == "load":
    pup = s.Pup(args)
  else:
    pup = ls.load(args["pickle"]) # Load object
    update(pup, )

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
