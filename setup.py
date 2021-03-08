# Copyright (c) 2021 Patricio Cubillos
# puppies is open-source software under the MIT license (see LICENSE)

import os
import sys
import re
import setuptools
from setuptools import setup, Extension

from numpy import get_include

sys.path.append(os.path.join(os.path.dirname(__file__), 'puppies'))
from VERSION import __version__


srcdir = 'src_c/'          # C-code source folder
incdir = 'src_c/include/'  # Include filder with header files

cfiles = os.listdir(srcdir)
cfiles = list(filter(lambda x: re.search('.+[.]c$', x), cfiles))
cfiles = list(filter(lambda x: not re.search('[.#].+[.]c$', x), cfiles))

inc = [get_include(), incdir]
eca = ['-ffast-math']
ela = []

extensions = [
    Extension(
        'puppies.lib.' + cfile.rstrip('.c'),
        sources=[f'{srcdir}{cfile}'],
        include_dirs=inc,
        extra_compile_args=eca,
        extra_link_args=ela)
    for cfile in cfiles
    ]


setup(
    name = 'puppies',
    version = __version__,
    author = 'Patricio Cubillos',
    author_email = 'patricio.cubillos@oeaw.ac.at',
    url = 'https://github.com/pcubillos/puppies',
    packages = setuptools.find_packages(),
    install_requires = [
        'numpy>=1.8.1',
        'scipy>=0.13.3',
        'matplotlib>=1.3.1',
        'astropy>=3.1',
        'mc3>=3.0.5',
        ],
    include_package_data=True,
    license = 'MIT',
    description = 'The Public Photometry Pipeline for Exoplanets',
    include_dirs = inc,
    ext_modules = extensions,
    )
