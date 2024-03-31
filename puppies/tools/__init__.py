# Copyright (c) 2021-2024 Patricio Cubillos
# puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

from .tools import *
from .parse import *
from .paramedit import *

__all__ = (
      tools.__all__
    + parse.__all__
    + paramedit.__all__
    )

# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__):
        del locals()[varname]
del(varname)

