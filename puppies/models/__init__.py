# Copyright (c) 2018 Patricio Cubillos and contributors.
# puppies is open-source software under the MIT license (see LICENSE).

from .mandeltr  import *
from .mandelecl import *
from .linramp   import *
from .quadramp  import *
from .expramp   import *
from .bliss     import *

from .mandeltr  import __all__ as mtall
from .mandelecl import __all__ as meall
from .linramp   import __all__ as lrall
from .quadramp  import __all__ as qrall
from .expramp   import __all__ as erall
from .bliss     import __all__ as biall

# Astrophysical:
__all__ = (meall + mtall
# Ramps:
           + lrall + qrall + erall
# Pixel maps:
           + biall)


# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__ ):
        del locals()[varname]
del(varname)

