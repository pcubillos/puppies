# Copyright (c) 2021-2024 Patricio Cubillos
# puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

import sys
import os

from puppies.tools import ROOT
sys.path.append(f'{ROOT}puppies/lib')
from _aphot import aphot

__all__ = [
    'aphot',
    'psf_binning',
    'position_to_index',
    'optimal_photometry',
]

def psf_binning(*args):
    """PSF routines"""
    raise NotImplementedError


def position_to_index(*args):
    """PSF routines"""
    raise NotImplementedError

def optimal_photometry(*args, **kwargs):
    """PSF routines"""
    raise NotImplementedError



# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__):
        del locals()[varname]
del(varname)

