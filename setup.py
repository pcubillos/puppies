# Copyright (c) 2021-2024 Patricio Cubillos
# puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

import os
import re
import setuptools
from setuptools import setup, Extension

from numpy import get_include


# Folders containing C source and header files:
srcdir = 'src_c/'
incdir = 'src_c/include/'

# Collect all .c files in src_dir:
cfiles = [
    c_file for c_file in os.listdir(srcdir)
    if c_file.endswith('.c') and not c_file.startswith('#')
]

inc = [get_include(), incdir]
eca = ['-ffast-math']
ela = []

extensions = [
    Extension(
        'puppies.lib.' + cfile.rstrip('.c'),
        sources=[f'{srcdir}{cfile}'],
        include_dirs=inc,
        extra_compile_args=eca,
        extra_link_args=ela,
    )
    for cfile in cfiles
]

setup(
    ext_modules = extensions,
    include_dirs = inc,
)
