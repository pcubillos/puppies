# Copyright (c) 2021 Patricio Cubillos
# puppies is open-source software under the MIT license (see LICENSE)

__all__ = [
    'linramp',
    'quadramp',
    'expramp',
    ]

import sys
import numpy as np

from ..tools import ROOT
sys.path.append(f"{ROOT}/puppies/lib")
import _linramp as lr
import _quadramp as qr
import _expramp as er


class Ramp(object):
    """Ramp-model superclass"""
    def __init__(self, time=None, mask=None, params=None):
        self.type = "ramp"
        self.npars = len(self.pnames)
        self.params = np.zeros(self.npars)
        if params is not None:
            self.params[:] = params
        self.pmin = np.tile(-np.inf, self.npars)
        self.pmax = np.tile( np.inf, self.npars)
        self.pstep = np.zeros(self.npars)
        if time is not None:
            self.setup(time)


    def __call__(self, params, time=None, mask=None):
        """
        Call function with self values.
        Update defaults if necessary.
        """
        if time is not None:
            self.time = time
        if mask is not None:
            self.mask = mask
        return self.eval(params, self.time[self.mask])


    def setup(self, time=None, mask=None, obj=None):
        """
        Set default independent variables (when calling eval without args).
        """
        if obj is not None:
            time = obj.time
            if mask is None:  # Input mask takes priority over pup.mask
                mask = obj.mask

        if mask is None:
            mask = np.ones(len(time), bool)

        self.time = time
        self.mask = mask



class linramp(Ramp):
    """
    Linear ramp model.
    Docstring me!

    Attributes
    ----------
    TBD

    Example
    -------
    TBD
    """
    def __init__(self, time=None, mask=None, params=None):
        self.name = "linramp"
        self.pnames = ["r1", "r0", "t0"]
        super(linramp, self).__init__(time, mask, params)

    def eval(self, params, time):
        """
        Evaluate the ramp function at the specified times.
        """
        return lr.linramp(params, time)


class quadramp(Ramp):
    """
    Quadratic ramp model.
    Docstring me!

    Attributes
    ----------
    TBD

    Example
    -------
    TBD
    """
    def __init__(self, time=None, mask=None, params=None):
        self.name = "quadramp"
        self.pnames = ["r2", "r1", "r0", "t0"]
        super(quadramp, self).__init__(time, mask, params)


    def eval(self, params, time):
        """
        Evaluate the ramp function at the specified times.
        """
        return qr.quadramp(params, time)


class expramp(Ramp):
    """
    Exponential ramp model.
    Docstring me!

    Attributes
    ----------
    TBD

    Example
    -------
    TBD
    """
    def __init__(self, time=None, mask=None, params=None):
        self.name = "expramp"
        self.pnames = ["goal", "r1", "r0", "pm"]
        super(expramp, self).__init__(time, mask, params)


    def eval(self, params, time):
        """
        Evaluate the ramp function at the specified times.
        """
        return er.expramp(params, time)
