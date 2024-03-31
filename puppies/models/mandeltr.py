import sys
import os
import numpy as np

from puppies.tools import ROOT
sys.path.append(f'{ROOT}puppies/lib')
import _mandeltr as mt


__all__ = ["mandeltr"]


class mandeltr():
  def __init__(self):
    self.name = "mandeltr"
    self.type = "astro"
    self.pnames = ["epoch", "rprs", "cosi", "ars", "flux", "per",
                   "c1", "c2", "c3", "c4"]
    self.npars = len(self.pnames)
    self.params = np.zeros(self.npars)
    self.pmin   = np.tile(-np.inf, self.npars)
    self.pmax   = np.tile( np.inf, self.npars)
    self.pstep  = np.zeros(self.npars)


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
    return mt.mandeltr(params, time)


  def setup(self, time=None, mask=None, pup=None):
    """
    Set independent variables and default values.
    """
    if pup is not None:
      time = pup.time
      if mask is None:  # Input mask takes priority over pup.mask
        mask = pup.mask

    if mask is None:
      mask = np.ones(len(time), bool)

    self.time = time
    self.mask = mask
