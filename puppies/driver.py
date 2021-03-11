# Copyright (c) 2021 Patricio Cubillos
# puppies is open-source software under the MIT license (see LICENSE)

import configparser

from . import inst
from . import tools as pt


def init(cfile):
    """
    Parse variables from a configuration file into a dictionary.
    """
    # Parse variables from a configuration file into a dictionary.
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read([cfile])

    if "PUPPIES" not in config.sections():
        pt.error(
            f"Invalid configuration file: '{cfile}', no [PUPPIES] section.")
    # Extract inputs:
    args = dict(config.items("PUPPIES"))

    # Check args contains "telescope":
    if "telescope" not in args.keys():
        pt.error(
            f"Invalid configuration file: '{cfile}', no 'telescope' parameter.")

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
