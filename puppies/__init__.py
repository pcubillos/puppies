# Copyright (c) 2021 Patricio Cubillos
# puppies is open-source software under the MIT license (see LICENSE)

__all__ = [
    'init',
    'core',
    'stats',
    'tools',
    'image',
    'io',
    'plots',
    'center',
    'photometry',
    'models',
    ]

# Import utility sub-packages:
from . import tools
from . import center
from . import photometry
from . import stats
from . import image
from . import io
from . import plots
from . import models
from .VERSION import __version__

# Initialization:
from .driver import init
# Core reduction modules:
from . import core


# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__):
        del locals()[varname]
del(varname)

