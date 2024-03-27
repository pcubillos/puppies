# Copyright (c) 2021-2024 Patricio Cubillos
# puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

from .pup_badpix import *
from .pup_center import *
from .pup_photom import *
from .pup_model  import *

__all__ = (
    pup_badpix.__all__
    + pup_center.__all__
    + pup_photom.__all__
    + pup_model.__all__
    )

# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__):
        del locals()[varname]
del(varname)

