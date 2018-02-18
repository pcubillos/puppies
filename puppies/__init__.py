# Copyright (c) 2018 Patricio Cubillos and contributors.
# puppies is open-source software under the MIT license (see LICENSE).

__all__ = ['init', 'badpix', 'cen', 'phot', 'setup', 'fit', 'mcmc',
           'stats', 'tools', 'io', 'image', 'plots', 'center', 'photometry',
           'models']

# Import utility sub-packages:
from . import tools
from . import center
from . import photometry
from . import stats
from . import io
from . import plots
from . import models

# Commands:
from .driver import init
from .pup_badpix import badpix
from .pup_center import driver as cen
from .pup_photom import driver as phot
from .pup_model  import setup, fit, mcmc

# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__ ):
        del locals()[varname]
del(varname)

