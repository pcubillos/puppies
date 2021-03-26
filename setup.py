# Copyright (c) 2021 Patricio Cubillos
# puppies is open-source software under the MIT license (see LICENSE)

import os
import sys
import re
from datetime import date
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

long_description = f"""
.. image:: https://raw.githubusercontent.com/pcubillos/puppies/aprilis/docs/figures/logo_puppies_texted.png
   :width: 50%

The Public Photometry Pipeline for Exoplanets

|Build Status|  |docs|  |PyPI|  |conda|  |License|

:copyright: Copyright {date.today().year} Patricio Cubillos
:license:   puppies is open-source software under the MIT license
:URL:       https://puppies.readthedocs.io/

.. |Build Status| image:: https://travis-ci.com/pcubillos/puppies.svg?branch=master
   :target: https://travis-ci.com/pcubillos/puppies

.. |docs| image:: https://readthedocs.org/projects/puppies/badge/?version=latest
    :target: https://puppies.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |PyPI| image:: https://img.shields.io/pypi/v/exo_puppies.svg
    :target: https://pypi.org/project/exo_puppies/
    :alt: Latest Version

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/puppies.svg
    :target: https://anaconda.org/conda-forge/puppies

.. |License| image:: https://img.shields.io/github/license/pcubillos/puppies.svg?color=blue
    :target: https://pcubillos.github.io/puppies.html
"""

setup(
    name = 'exo_puppies',
    version = __version__,
    author = 'Patricio Cubillos',
    author_email = 'pcubillos@fulbrightmail.org',
    url = 'https://github.com/pcubillos/puppies',
    packages = setuptools.find_packages(),
    install_requires = [
        'numpy>=1.8.1',
        'scipy>=0.13.3',
        'matplotlib>=1.3.1',
        'astropy>=3.1',
        'mc3>=3.0.6',
        ],
    include_package_data=True,
    license = 'MIT',
    description = 'The Public Photometry Pipeline for Exoplanets',
    include_dirs = inc,
    long_description = long_description,
    long_description_content_type="text/x-rst",
    entry_points={'console_scripts': ['pup = puppies.__main__:main']},
    ext_modules = extensions,
    )
