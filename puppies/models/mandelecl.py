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

  def __call__(self, params, x):
    return self.eval(params, x)

  def eval(self, params, x):
    return me.mandelecl(params, x)

