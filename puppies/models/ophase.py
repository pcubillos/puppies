import sys
import os
import numpy as np

topdir = os.path.realpath(os.path.dirname(__file__) + "/../..")
sys.path.append(topdir + "/puppies/lib")
import _mandeltr as mt

sys.path.append(topdir + "/modules/eclipse/eclipse/lib")
import _eclipse as ecl

__all__ = ["orbit"]


def orbital_phase(params, time):
    t_sec, per, cosi, A, B, C, D, r2, r2off = params
    t    = time - t_sec
    freq = 2*np.pi/per
    phi  = freq*t

    # calculate the phase variations
    if C==0 and D==0:
        #Skip multiplying by a bunch of zeros to speed up fitting
        ophase = 1 + A*(np.cos(phi)-1) + B*np.sin(phi)
    else:
        ophase = 1 + A*(np.cos(phi)-1)   + B*np.sin(phi)  \
                   + C*(np.cos(2*phi)-1) + D*np.sin(2*phi)
    if r2 != 1.0:
      # calculate the orbital phase (assumes the planet is tidally locked)
      phi = (freq*t-np.pi)%(2*np.pi)
      # effectively rotate the elongated axis by changing phi
      phi -= r2off*np.pi/180.0
      # convert inclination to radians:
      sini = np.sqrt(1-cosi**2)

      ophase *= np.sqrt(sini**2 * (r2**2*np.sin(phi)**2 + np.cos(phi)**2)
                      + cosi**2*r2**2)

    # Phase flux cannot be negative:
    #if np.any(ophase < 0):
    #  ophase[:] = 1e100
    return ophase


class orbit():
  def __init__(self):
    self.name = "orbit"
    self.type = "astro"
    self.pnames = ["epoch", "rprs", "cosi", "ars", "flux", "per",
                   "c1", "c2", "c3", "c4",
                   "midpt", "width", "depth", "ting", "tegr",
                   "A", "B", "C", "D", "r2ratio", "r2offset"]
    self.npars = len(self.pnames)
    self.params = np.zeros(self.npars)
    self.pmin   = np.tile(-np.inf, self.npars)
    self.pmax   = np.tile( np.inf, self.npars)
    self.pstep  = np.zeros(self.npars)

    # Indices for each component:
    idx = np.arange(len(self.pnames))
    self.itransit = idx < 10
    self.ieclipse = (idx >= 10) & (idx<15)
    self.iphase   = idx >= 15
    self.eparams = np.ones(6)
    self.oparams = np.zeros(9)

  def __call__(self, params, time=None, mask=None):
    """
    Call function with self vaues.
    Update defaults if necessary.
    """
    if time is not None:
      self.time = time
    if mask is not None:
      self.mask = mask
    return self.eval(params, self.time[self.mask])


  def eval(self, params, time=None):
    """
    Evaluate function at specified input times.
    """
    # Star+transit:
    fstar = mt.mandeltr(params[self.itransit], self.time[self.mask])
    # Eclipses:
    self.eparams[0:5] = params[self.ieclipse]
    self.eparams[5] = params[4]   # Flux
    fplanet = ecl.mandelecl(self.eparams, self.time[self.mask])
    # Add second eclipse:
    self.eparams[0] += params[5]  # period
    fplanet += ecl.mandelecl(self.eparams, self.time[self.mask])
    fplanet -= self.eparams[5]*(2 + params[12])
    #fplanet *= params[4]

    # Orbital phase:
    self.oparams[0:3] = params[10], params[5], params[2]
    self.oparams[3:]  = params[self.iphase]
    ophase = orbital_phase(self.oparams, self.time[self.mask])

    return fstar + fplanet*ophase


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

