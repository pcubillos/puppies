# Copyright (c) 2018-2019 Patricio Cubillos and contributors.
# puppies is open-source software under the MIT license (see LICENSE).

import os
import sys
import re
import setuptools
from setuptools import setup, Extension

from numpy import get_include

topdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(topdir + "/puppies")
import VERSION as ver

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


setup(
    name = "puppies",
    version = "{ver.pup_VER}.{ver.pup_MIN}.{ver.pup_REV}"
    description = "The Public Photometry Pipeline for Exoplanets.",
    author = "Patricio Cubillos",
    author_email = "patricio.cubillos@oeaw.ac.at",
    url = "https://github.com/pcubillos/puppies",
    packages = setuptools.find_packages(),
    install_requires = [
        'numpy>=1.8.1',
        'scipy>=0.13.3',
        'matplotlib>=1.3.1',
        'astropy>=3.1',
        ],
    license = ["MIT"],
    include_dirs = inc,
    ext_modules  = extensions)
