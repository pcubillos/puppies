# Copyright (c) 2018 Patricio Cubillos and contributors.
# puppies is open-source software under the MIT license (see LICENSE).

import os
import sys

from .mandeltr  import *
from .ramps     import *
from .bliss     import *
sys.path.append(os.path.dirname(os.path.realpath(__file__))
                + "/../../modules/eclipse/")
from eclipse import *

from .mandeltr  import __all__ as mtall
from eclipse    import __all__ as eall
from .ramps     import __all__ as rall
from .bliss     import __all__ as biall

# Astrophysical:
__all__ = (mtall + eall
# Ramps:
         + rall
# Pixel maps:
         + biall)


# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__ ):
        del locals()[varname]
del(varname)

