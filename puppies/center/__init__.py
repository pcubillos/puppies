# Copyright (c) 2018 Patricio Cubillos and contributors.
# puppies is open-source software under the MIT license (see LICENSE).

from .gaussian import *
#from .sigrej    import *

from .gaussian import __all__ as gall
#from .sigrej    import __all__
#stats_all += __all__

__all__ = gall


# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__ ):
        del locals()[varname]
del(varname)

