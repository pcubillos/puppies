import sys
import os
import numpy as np

topdir = os.path.realpath(os.path.dirname(__file__) + "/../..")
sys.path.append(topdir + "/puppies/lib")
import _linramp as lr


__all__ = ["linramp"]

class linramp():
  def __init__(self):
    self.name = "linramp"
    self.type = "ramp"
    self.pnames = ["r1", "r0", "t0"]
    self.npars  = len(self.pnames)
    self.params = np.zeros(self.npars)
    self.pmin   = np.tile(-np.inf, self.npars)
    self.pmax   = np.tile( np.inf, self.npars)
    self.pstep  = np.zeros(self.npars)

  def __call__(self, params, x):
    return self.eval(params, x)

  def eval(self, params, x):
    return lr.linramp(params, x)

