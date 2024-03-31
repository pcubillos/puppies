.. puppies documentation master file, created by
   sphinx-quickstart on Sat May 27 21:39:37 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

puppies
=======

The Public Photometry Pipeline for Exoplanets
---------------------------------------------

|Build Status|  |docs|  |PyPI|  |conda|  |License|

-------------------------------------------------------------------

:Author:       Patricio Cubillos
:Contact:      `pcubillos[at]fulbrightmail.org`_
:Organizations: `Space Research Institute (IWF)`_
:Web Site:     https://github.com/pcubillos/puppies
:Date:         |today|

Features
========

**puppies** is a general Python library to reduce Spitzer, HST, and
JWST time-series observations of exoplanets (you could say, it's a
general TSO type of tool).  The **puppies** pipeline includes:

1. reduction of the 2D frames to extract the raw light curves, and
2. light-curve analysis including a variety of
   astrophysical modes (transits, eclipses, and phase curves) and
   telescope systematics (time- and pointing-dependent).

.. _team:

Contributors
============

- `Patricio Cubillos`_ (IWF) `pcubillos[at]fulbrightmail.org`_

Documentation
=============

.. toctree::
   :maxdepth: 2

   install
   pup_marker
   WASP18b_eclipse
   contributing
   license

Be Kind
=======

Please reference this paper if you found ``puppies`` useful for your research:
  `Cubillos et al. (2024): The Public Photometry Pipeline for Exoplanets <https://www.youtube.com/watch?v=dQw4w9WgXcQ?autoplay=1>`_, Prima Aprilis.

We welcome your feedback, but do not necessarily guarantee support.
Please send feedback or inquiries to:

  Patricio Cubillos (`pcubillos[at]fulbrightmail.org`_)

``puppies`` is open-source open-development software under the GNU GPL v2 :ref:`license`.

Thank you for using ``puppies``!


.. _Patricio Cubillos: https://github.com/pcubillos/
.. _pcubillos[at]fulbrightmail.org: pcubillos@fulbrightmail.org
.. _Space Research Institute (IWF): http://iwf.oeaw.ac.at/


.. |Build Status| image:: https://github.com/pcubillos/bibmanager/actions/workflows/python-package.yml/badge.svg
   :target: https://github.com/pcubillos/bibmanager/actions/workflows/python-package.yml

.. |docs| image:: https://readthedocs.org/projects/puppies/badge/?version=latest
    :target: https://puppies.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |PyPI| image:: https://img.shields.io/pypi/v/exo_puppies.svg
    :target: https://pypi.org/project/exo_puppies/
    :alt: Latest Version

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/exo_puppies.svg
    :target: https://anaconda.org/conda-forge/exo_puppies

.. |License| image:: https://img.shields.io/github/license/pcubillos/puppies?color=blue
   :alt: GitHub License
    :target: https://puppies.readthedocs.io/en/latest/license.html

