import sys
import os
import numpy as np

topdir = os.path.realpath(os.path.dirname(__file__) + "/../..")
sys.path.append(topdir + "/puppies/lib")
import _expramp as er


__all__ = ["expramp"]

class expramp():
  def __init__(self):
    self.name = "expramp"
    self.type = "ramp"
    self.pnames = ["goal", "r1", "r0", "pm"]
    self.npars  = len(self.pnames)
    self.params = np.array([1.0, 1.0, 0.5, -1.0])
    self.pmin   = np.tile(-np.inf, self.npars)
    self.pmax   = np.tile( np.inf, self.npars)
    self.pstep  = np.zeros(self.npars)


  def __call__(self, params, time=None, mask=None):
    """
    Call function with self vaues.
    Update if necessary.
    """
    if time is not None:
      self.time = time
    if mask is not None:
      self.mask = mask
    return self.eval(params, self.time[self.mask])


  def eval(self, params, time):
    """
    Evaluate function at specified input times.
    """
    return er.expramp(params, time)


  def setup(self, time=None, mask=None, pup=None):
    """
    Set default independent variables (when calling eval without args).
    """
    if pup is not None:
      time = pup.time
    if mask is None:
      mask = np.ones(len(time), bool)
    self.time = time
    self.mask = mask
