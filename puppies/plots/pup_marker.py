# Copyright (c) 2021-2024 Patricio Cubillos
# puppies is open-source software under the MIT license (see LICENSE)

from packaging import version

import numpy as np
import matplotlib
import matplotlib.pyplot
from matplotlib.transforms import Affine2D
from matplotlib.path import Path


def make_pup_path():
    codes = [
        Path.MOVETO, Path.LINETO, Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.MOVETO,
        ]
    verts = np.array([
        [136.02323161,  11.05746567],
        [129.5140526 ,  -7.58628508],
        [194.50664877,  -7.58628508],
        [199.48322077,  28.67487101],
        [209.8192248 ,  53.77659507],
        [244.51866689,  86.26117916],
        [251.16324091,  99.55032719],
        [244.51866689, 133.90035236],
        [236.39752087, 136.85349637],
        [226.79980284, 142.75978439],
        [208.3426528 , 102.15405428],
        [183.97921473, 111.75177231],
        [164.78377868, 125.04092034],
        [131.5609086 ,  85.91176224],
        [125.65462058,  65.97804019],
        [118.21548822,  32.74651275],
        [118.18140072,  15.02231697],
        [139.19511266,  12.24692106],
        [158.62288407,  13.83286158],
        [169.68253968,  23.60750361],
        [176.41654642,  46.6955267 ],
        [181.77005224,  10.78613284]])
    v = verts - np.mean(verts,0)
    v /= np.ptp(v)
    pup_path = Path(4*v, codes, closed=False)
    return pup_path



def _set_pup(self):
    """A puppy marker for matplotlib."""
    # Not totally sure of what's going on here exactly, just following
    # what's in matplotlib/markers.py  for def _set_pentagon()
    self._transform = Affine2D().scale(0.5)
    self._snap_threshold = 5.0
    self._filled = False
    self._path = make_pup_path()

    if version.parse(matplotlib.__version__) >= version.parse('3.4'):
        from matplotlib._enums import JoinStyle
        self._joinstyle = self._user_joinstyle or JoinStyle.miter
    else:
        self._joinstyle = 'miter'


def _set_kitten(self):
    pass


# Add the marker style to the matplotlib list of markers:
matplotlib.markers.MarkerStyle._set_pup = _set_pup
matplotlib.markers.MarkerStyle.markers["pup"] = "pup"
#matplotlib.markers.MarkerStyle._set_kitten = _set_kitten
#matplotlib.markers.MarkerStyle.markers["kitten"] = "kitten"


#def test():
#    fs = 'full'
#    ms = 8
#    mfc = None
#
#    plt.figure(0)
#    plt.clf()
#    plt.plot([0], [1], marker='kitten', mfc=mfc, ms=ms, fillstyle=fs)
#    plt.plot([0], [0], marker='kitten', ms=ms, fillstyle=None)
#    plt.plot([1], [1], marker='s', mfc=mfc, ms=ms, fillstyle=fs)
#    plt.plot([0.5], [1], marker='*', mfc=mfc, ms=ms, fillstyle=fs)
#    plt.xlim(-2, 3)
#    plt.ylim(-2, 2)
#    #plt.plot([0], [1], marker='pup')




