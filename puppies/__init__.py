# Copyright (c) 2021-2024 Patricio Cubillos
# puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

__all__ = [
    'center',
    'core',
    'image',
    'io',
    'models',
    'photometry',
    'plots',
    'stats',
    'tools',
]

from . import center
from . import core
from . import image
from . import io
from . import models
from . import photometry
from . import plots
from . import stats
from . import tools
from .version import __version__



# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__):
        del locals()[varname]
del(varname)

