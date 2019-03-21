import os
import sys
import re
import setuptools
from setuptools import setup, Extension

from numpy import get_include

topdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(topdir + "/puppies")
import VERSION as ver
__version__ = f"{ver.PUP_VER}.{ver.PUP_MIN}.{ver.PUP_REV}"

srcdir = topdir + '/src_c/'          # C-code source folder
incdir = topdir + '/src_c/include/'  # Include filder with header files

files = os.listdir(srcdir)
# This will filter the results for just the c files:
files = list(filter(lambda x:     re.search('.+[.]c$',     x), files))
files = list(filter(lambda x: not re.search('[.#].+[.]c$', x), files))

inc = [get_include(), incdir]
eca = ['-ffast-math']
ela = []

extensions = []
for efile in files:
    e = Extension('puppies.lib.'+efile.rstrip(".c"),
                  sources=["{:s}{:s}".format(srcdir, efile)],
                  include_dirs=inc,
                  extra_compile_args=eca,
                  extra_link_args=ela)
    extensions.append(e)


setup(name         = "puppies",
      version      = __version__,
      author       = "Patricio Cubillos",
      author_email = "patricio.cubillos@oeaw.ac.at",
      url          = "https://github.com/pcubillos/puppies",
      packages     = setuptools.find_packages(),
      install_requires = ['numpy>=1.8.1',
                          'scipy>=0.13.3',
                          'matplotlib>=1.3.1'],
      license      = ["MIT"],
      description  = "Public Photometry Pipeline for Exoplanets.",
      include_dirs = inc,
      ext_modules  = extensions)
