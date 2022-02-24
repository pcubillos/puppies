# Copyright (c) 2021 Patricio Cubillos
# puppies is open-source software under the MIT license (see LICENSE)

import sys
import os

from puppies.tools import ROOT
sys.path.append(f'{ROOT}puppies/lib')
# Debugging:
#from aphot import aphot

__all__ = ["aphot"]


# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__):
        del locals()[varname]
del(varname)

