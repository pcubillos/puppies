# Copyright (c) 2018 Patricio Cubillos and contributors.
# puppies is open-source software under the MIT license (see LICENSE).

__all__ = [
    'init',
    'core',
    'stats',
    'tools',
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
from . import io
from . import plots
from . import models

# Initialization:
from .driver import init
# Core reduction modules:
from . import core

__version__ = f"{ver.PUP_VER}.{ver.PUP_MIN}.{ver.PUP_REV}"

# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__):
        del locals()[varname]
del(varname)

