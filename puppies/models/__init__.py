# Copyright (c) 2018 Patricio Cubillos and contributors.
# puppies is open-source software under the MIT license (see LICENSE).

from .linramp   import *
from .mandelecl import *
from .bliss     import *

from .linramp   import __all__ as lrall
from .mandelecl import __all__ as meall
from .bliss     import __all__ as biall

__all__ = lrall + meall + biall


# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__ ):
        del locals()[varname]
del(varname)

