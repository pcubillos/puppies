# Copyright (c) 2021-2024 Patricio Cubillos
# puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

__all__ = [
    "eclipse",
    "mandelecl",
]


import os
import sys
import numpy as np

from puppies.tools import ROOT
sys.path.append(f'{ROOT}puppies/lib')
import _eclipse as ecl


class Ecl(object):
  """
  Eclipse-model superclass.
  """
  def __init__(self, time=None, mask=None, params=None):
    self.type = "astro"
    self.npars = len(self.pnames)
    self.params = np.zeros(self.npars)
    if params is not None:
      self.params[:] = params
    self.pmin   = np.tile(-np.inf, self.npars)
    self.pmax   = np.tile( np.inf, self.npars)
    self.pstep  = np.zeros(self.npars)
    if time is not None:
      self.setup(time)


  def __call__(self, params, time=None, mask=None, update=False):
    """
    Call function using self values as defaults.

    Parameters
    ----------
    params: 1D float ndarray
       Model parameters (see class docstring for details).
    time: 1D float ndarray
       Phase/times where to evaluate the model.
    mask: 1D bool ndarray
       Times good-value mask.
    update: Bool
       If True, updated the object's attributes.
    """
    if update:
      self.setup(time, mask, params)

    if time is None:
      time = self.time

    if mask is not None:
      return self.eval(params, time[mask])
    elif self.mask is not None:
      return self.eval(params, time[self.mask])

    return self.eval(params, time)


  def setup(self, time=None, mask=None, params=None, obj=None):
    """
    Set the model's attributes (when not None).

    Parameters
    ----------
    time: 1D float ndarray
       Phase/times where to evaluate the model.
    mask: 1D bool ndarray
       Times good-value mask.
    params: 1D float ndarray
       Model parameters.
    obj: An object
       If not None, extract the time and mask values from the
       object's attributes.
    """
    if obj is not None:
      time = obj.time
      if mask is None:  # Input mask takes priority over obj.mask
        mask = obj.mask
      # FINDME: do I want params?

    # Independent variables:
    if time is not None:
      if mask is None:
        mask = np.ones(len(time), bool)
      self.time = time
      self.mask = mask
    # Model parameters:
    if params is not None:
      self.params[:] = params


class eclipse(Ecl):
  """
  A secondary-eclipse model for the times of the Webb.

  This class implements a Mandel & Agol eclipse model with independent
  ingress and egress depths, and the out-of-eclipse flux as a second-
  degree polynomial.

  This model has nine parameters (self.params, see also self.pnames):
   - midpt: Mid-eclipse epoch.
   - width: Eclipse duration between first and fourth contacts (T14).
   - idepth: Normalized eclipse depth at ingress (T1) relative to a
             stellar flux of 1.0.  That is,
             idepth = (flux(T1) - flux(T2))/flux(T2).
   - edepth: Normalized eclipse depth at egress (T4) relative to a
             stellar flux of 1.0.  That is,
             edepth = (flux(T4) - flux(T3))/flux(T3).
   - ting: Ingress duration (time between first and second contacts, T12).
   - tegr: Egress duration (time between third and fourth contacts, T34).
   - flux: Stellar flux level.  That is, flux during eclipse:
           flux = flux(T2) = flux(T3).
   - slope: Out-of-eclipse linear slope.
   - quad: Out-of-eclipse quadratic term slope.

  The time units are arbitrary, and thus, a user's choice.  Note that
  midpt, width, ting, terg, and time must have consistent units.

  Attributes
  ----------
  name: String
     The model's name.
  type: String
     The type of model.
  pnames: 1D string list
     Names of the model parameters.
  npars: Integer
     The number of model parameters.
  params: 1D float ndarray
     The model parameter values.
  pmin: 1D float ndarray
     Minimum-value boundary of the parameters (for MCMCs).
  pmax: 1D float ndarray
     Maximum-value boundary of the parameters (for MCMCs).
  pstep: 1D float ndarray
     Parameter stepsize (for MCMCs).
  time: 1D float ndarray
     The model's independent variable (timestamps where to evaluate).
  mask: 1D bool ndarray
     Time good-value mask.

  Example
  -------
  >>> import sys
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt

  >>> sys.path.append("../eclipse")
  >>> import eclipse as ecl

  >>> # Create an eclipse model with given orbital-phase timestamps:
  >>> phase = np.linspace(0.35, 0.65, 300)
  >>> model = ecl.eclipse(time=phase)
  >>> # Define eclipse parameters:
  >>> #               [midpt width idepth edepth ting  tegr  flux slope  quad]
  >>> params = np.array([0.5, 0.1, 0.01, 0.008,  0.01, 0.01, 1.0, -0.02, -0.03])
  >>> # One can call the eval function:
  >>> eclipse1 = model.eval(params, phase)
  >>> # Or directly call the object (no need to pass the phase):
  >>> params[8] = 0.0  # Linear-slope model
  >>> eclipse2 = model(params)

  >>> # Show results:
  >>> plt.figure(0)
  >>> plt.clf()
  >>> plt.plot(phase, eclipse1, ".-", color='b')
  >>> plt.plot(phase, eclipse2, ".-", color='orange')
  >>> plt.xlim(0.34, 0.66)
  >>> plt.ylim(0.999, 1.013)
  >>> plt.xlabel("Orbital phase")
  >>> plt.ylabel("Flux")
  """
  def __init__(self, time=None, mask=None, params=None):
    """
    Class constructor.

    Parameters
    ----------
    time: 1D float ndarray
       Phase/times where to evaluate the model.
    mask: 1D bool ndarray
       Times good-value mask.
    params: 1D float ndarray
       Model parameters (see class docstring for details).
    """
    self.name = "eclipse"
    self.pnames = ["midpt", "width", "idepth", "edepth", "ting", "tegr",
                   "flux", "slope", "quad"]
    super(eclipse, self).__init__(time, mask, params)
    # Update pmin:
    self.pmin = np.array([-np.inf, 0, 0, 0, 0, 0, 0, -np.inf, -np.inf])


  def eval(self, params, time):
    """
    Evaluate the eclipse function at the specified times.

    Parameters
    ----------
    params: 1D float ndarray
       Model parameters (see class docstring for details).
    time: 1D float ndarray
       Phase/times where to evaluate the model.
    """
    return ecl.eclipse_quad(params, time)


class mandelecl(Ecl):
  """
  Secondary-eclipse model from Mandel & Agol (2002).

  This model has six parameters (self.params, see also self.pnames):
   - midpt: mid-eclipse epoch.
   - width: Eclipse duration between first and fourth contacts (T14).
   - depth: Normalized eclipse depth relative to a stellar flux of 1.0.
            That is, depth = dflux/flux.
   - ting:  Ingress duration (time between first and second contacts, T12).
   - tegr:  Egress duration (time between third and fourth contacts, T34).
   - flux:  Stellar flux level, i.e., flux during eclipse.

  The time units are arbitrary, and thus, a user's choice.  Note that
  midpt, width, ting, terg, and time must have consistent units.

  Attributes
  ----------
  name: String
     The model's name.
  type: String
     The type of model.
  pnames: 1D string list
     Names of the model parameters.
  npars: Integer
     The number of model parameters.
  params: 1D float ndarray
     The model parameter values.
  pmin: 1D float ndarray
     Minimum-value boundary of the parameters (for MCMCs).
  pmax: 1D float ndarray
     Maximum-value boundary of the parameters (for MCMCs).
  pstep: 1D float ndarray
     Parameter stepsize (for MCMCs).
  time: 1D float ndarray
     The model's independent variable (timestamps where to evaluate).
  mask: 1D bool ndarray
     Time good-value mask.

  Example
  -------
  >>> import sys
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt

  >>> sys.path.append("../eclipse")
  >>> import eclipse as ecl

  >>> # Create a mandelecl model (setting model.time):
  >>> phase = np.linspace(0.35, 0.65, 300)
  >>> model = ecl.mandelecl(phase)
  >>> # Define eclipse parameters:
  >>> #                 midpt width depth ting  tegr  flux
  >>> params = np.array([0.5, 0.1,  0.01, 0.01, 0.01, 1.0])
  >>> # One can call the eval function:
  >>> eclipse1 = model.eval(params, phase)
  >>> # Or directly call the object:
  >>> params[2] = 0.015  # Change eclipse depth
  >>> eclipse2 = model(params, phase)
  >>> # This call can use the default phase, so no need to pass as argument:
  >>> params[5] = 1.005  # Change flux level
  >>> eclipse3 = model(params)

  >>> # Show results:
  >>> plt.figure(0)
  >>> plt.clf()
  >>> plt.plot(phase, eclipse1, ".-", color='b')
  >>> plt.plot(phase, eclipse2, ".-", color='orange')
  >>> plt.plot(phase, eclipse3, ".-", color='limegreen')
  >>> plt.xlim(0.34, 0.66)
  >>> plt.xlabel("Orbital phase")
  >>> plt.ylabel("Flux")
  """
  def __init__(self, time=None, mask=None, params=None):
    """
    Class constructor.

    Parameters
    ----------
    time: 1D float ndarray
       Phase/times where to evaluate the model.
    mask: 1D bool ndarray
       Times good-value mask.
    params: 1D float ndarray
       Model parameters (see class docstring for details).
    """
    self.name = "mandelecl"
    self.pnames = ["midpt",  "width",  "depth",  "ting",  "tegr",  "flux"]
    super(mandelecl, self).__init__(time, mask, params)
    # Update pmin:
    self.pmin   = np.array([-np.inf, 0.0, 0.0, 0.0, 0.0, -np.inf])


  def eval(self, params, time):
    """
    Evaluate the eclipse function at the specified times.

    Parameters
    ----------
    params: 1D float ndarray
       Model parameters (see class docstring for details).
    time: 1D float ndarray
       Phase/times where to evaluate the model.
    """
    return ecl.mandelecl(params, time)
