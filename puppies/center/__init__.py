# Copyright (c) 2018 Patricio Cubillos and contributors.
# puppies is open-source software under the MIT license (see LICENSE).

from . import gaussian
from .least_asym import *
from .col        import *

from .least_asym import __all__ as lall
from .col        import __all__ as call

__all__ = ["gaussian"] + lall + call


# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__ ):
        del locals()[varname]
del(varname)

