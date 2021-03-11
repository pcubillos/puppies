# Copyright (c) 2021 Patricio Cubillos
# puppies is open-source software under the MIT license (see LICENSE)

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
    pt.msg(1, f"Updated user parameters: {list(args.keys())}")

    # Set defaults for centering parameters:
    pup.inputs.setdefault("ncpu", "1")
    pup.inputs.setdefault("ctrim", "8")
    pup.inputs.setdefault("fitbg", "True")
    pup.inputs.setdefault("cweights", "False")
    pup.inputs.setdefault("aradius", "0")
    pup.inputs.setdefault("asize", "0")
    pup.inputs.setdefault("psftrim", "0")
    pup.inputs.setdefault("psfarad", "0")
    pup.inputs.setdefault("psfasize", "0")
    pup.inputs.setdefault("psfscale", "0")

    # Check all necessary inputs are provided:
    if "centering" not in pup.inputs.keys():
        pt.error("Missing 'centering' user input.")

    pup.centering = pt.parray(pup.inputs["centering"])
    pup.ncpu = int( pup.inputs["ncpu"])
    pup.ctrim = int( pup.inputs["ctrim"])
    pup.cweights = bool(pup.inputs["cweights"])
    pup.fitbg = bool(pup.inputs["fitbg"])
    pup.aradius = int( pup.inputs["aradius"])
    pup.asize = int( pup.inputs["asize"])
    pup.psftrim = int( pup.inputs["psftrim"])
    pup.psfarad = int( pup.inputs["psfarad"])
    pup.psfasize = int( pup.inputs["psfasize"])
    pup.psfscale = int( pup.inputs["psfscale"])

    if "lag" in pup.centering:
        if pup.aradius == 0 or pup.asize == 0:
            pt.error("Missing 'aradius' or 'asize' least-asymmetry inputs.")
        if os.path.isfile(pup.psf):
            if pup.psfarad == 0 or pup.psfasize == 0:
                pt.error("Missing 'psfaradius' or 'psfasize' least-asymmetry inputs.")

    if "psffit" in pup.centering:
        if pup.psfscale == 0:
            pt.error("Missing 'psfscale' centering user input.")

