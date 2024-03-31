# Copyright (c) 2021-2024 Patricio Cubillos
# puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

__all__ = [
    'save',
    'load',
    'update',
    ]

import os
import configparser
import pickle

import numpy as np

from .. import tools as pt


def save(pup):
    """
    Save object into pickle file keeping specified variables into a
    separate npz file.
    """
    # List of variable to be saved into npz file:
    varnames = ["data", "uncert", "mask", "head", "bdmskd", "brmskd"]
    # Output npz file:
    savefile = f"{pup.folder}/{pup.ID}.npz"
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
    with open(f"{pup.folder}/{pup.ID}.p", "wb") as f:
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
            #idx = f.files.index(param)
            val = f[param]
        return val


def update(pup, cfile):
    """
    Update a puppies object with parameters from the input config file.
    """
    # Extract inputs:
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read([cfile])
    args = dict(config.items("PUPPIES"))

    pup.inputs.update(args)
    pt.msg(1, f"Updated parameters: {list(args.keys())}")

    # I don't really need this, don't I?:
    inputs = {
        "ncpu": "1",
        # Centering:
        "ctrim": "8",
        "fitbg": "True",
        "cweights": "False",
        "aradius": "0",
        "asize": "0",
        "psftrim": "0",
        "psfarad": "0",
        "psfasize": "0",
        "psfscale": "0",
        # Photometry:
        "skyfrac": "0.1",
        "skymed": "False",
        "expand": "1",
        "psfexpand": "1",
        "otrim": "10",
    }

    # General:
    if 'ncpu' in args:
        pup.ncpu = int(args["ncpu"])

    # Centering:
    if 'centering' in args:
        pup.centering = pt.parray(args["centering"])
    if 'ctrim' in args:
        pup.ctrim = int(args["ctrim"])
    if 'cweights' in args:
        pup.cweights = bool(args["cweights"])
    if 'fitbg' in args:
        pup.fitbg = bool(args["fitbg"])
    if 'aradius' in args:
        pup.aradius = int(args["aradius"])
    if 'asize' in args:
        pup.asize = int(args["asize"])
    if 'psftrim' in args:
        pup.psftrim = int(args["psftrim"])
    if 'psfarad' in args:
        pup.psfarad = int(args["psfarad"])
    if 'psfasize' in args:
        pup.psfasize = int(args["psfasize"])
    if 'psfscale' in args:
        pup.psfscale = int(args["psfscale"])

    # Photometry:
    if 'photap' in args:
        pup.photap = pt.parray(args["photap"])
    if 'skyin' in args:
        pup.skyin = pt.parray(args["skyin"], float)
    if 'skyout' in args:
        pup.skyout = pt.parray(args["skyout"], float)
    if 'ncpu' in args:
        pup.ncpu = int(args["ncpu"])
    if 'skyfrac' in args:
        pup.skyfrac = float(args["skyfrac"])
    if 'skymed' in args:
        pup.skymed = bool(args["skymed"])
    if 'expand' in args:
        pup.expand = int(args["expand"])
    if 'psfexpand' in args:
        pup.psfexpand = int(args["psfexpand"])
    if 'otrim' in args:
        pup.otrim = int(args["otrim"])

