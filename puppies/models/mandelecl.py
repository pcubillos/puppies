import sys
import os
import numpy as np

topdir = os.path.realpath(os.path.dirname(__file__) + "/../..")
sys.path.append(topdir + "/puppies/lib")
import _mandelecl as me


__all__ = ["mandelecl"]

class mandelecl():
  def __init__(self):
    self.name = "mandelecl"
    self.type = "astro"
    self.pnames = ["midpt",  "width",  "depth",  "tin",  "teg",  "flux"]
    self.npars = len(self.pnames)
    self.params = np.zeros(self.npars)
    self.pmin   = np.array([-np.inf, 0.0, 0.0, 0.0, 0.0, -np.inf])
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
    return me.mandelecl(params, time)


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
