# Copyright (c) 2021-2022 Patricio Cubillos
# puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

__all__ = [
    'init',
    'run',
]

import configparser

from . import instruments as inst
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
    elif args["telescope"] == "hst":
        pup = None
    elif args["telescope"] == "jwst":
        pup = None

    return pup


def run(args):
    """
    Hello, this is doc.
    """
    runmode = pt.parray(args["runmode"])
    nsteps = len(runmode)

    if runmode[0] == "load":
        pup = s.Pup(args)
    else:
        pup = io.load(args["pickle"])
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
