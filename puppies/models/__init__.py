# Copyright (c) 2021 Patricio Cubillos
# puppies is open-source software under the MIT license (see LICENSE)

import os
import sys

from .mandeltr import *
from .ophase import *
from .ramps import *
#from .aor import *
from .bliss import *
from .eclipse import *

from .mandeltr import __all__ as mtall
from .ophase import __all__ as oall
from .eclipse import __all__ as eall
from .ramps import __all__ as rall
#from .aor import __all__ as aall
from .bliss import __all__ as biall

__all__ = (
    # Astrophysical:
    mtall
    + eall
    + oall
    # Ramps:
    + rall
    #+ aall
    # Pixel maps:
    + biall
    )


# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__):
        del locals()[varname]
del(varname)

