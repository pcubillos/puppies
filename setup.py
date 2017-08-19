import os, re, sys
from numpy import get_include
from setuptools import setup, Extension

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
eca = []
ela = []

extensions = []
for i in range(len(files)):
  e = Extension(files[i].rstrip(".c"),
                sources=["{:s}{:s}".format(srcdir, files[i])],
                include_dirs=inc,
                extra_compile_args=eca,
                extra_link_args=ela)
  extensions.append(e)


setup(name         = "puppies",
      version      = "{:d}.{:d}.{:d}".format(ver.pup_VER, ver.pup_MIN,
                                             ver.pup_REV),
      author       = "Patricio Cubillos",
      author_email = "patricio.cubillos@oeaw.ac.at",
      url          = "https://github.com/pcubillos/puppies",
      packages     = ["puppies"],
      license      = ["MIT"],
      description  = "Public Photometry Pipeline for Exoplanets.",
      include_dirs = inc,
      #scripts      = ['MCcubed/mccubed.py'],
      #entry_points={"console_scripts": ['foo = MCcubed.mccubed:main']},
      ext_modules  = extensions)
