.. _wasp43b_eclipse:

Worked Example: WASP-43b Eclipse
================================

The following worked example shows the steps to reduce raw Spitzer
time-series data of a secondary eclipse of WASP-43b.  The first
section of this example will produce several raw light curves under
different centering and photometry configurations.

The second section will fit instrumental+astrophysical models to the
raw lightcurves, determine the optimal reduction configuration and
determine the optimal modeling configuration.


Download Spizer Data
--------------------

Follow these steps on the browser to download a Spitzer/IRAC2 (4.5 um)
time-series during a secondary-eclipse of WASP-43b:

.. code-block:: shell

  # - Go to this site on the browser:
  #   https://sha.ipac.caltech.edu/applications/Spitzer/SHA/

  # - On the left menu, click on 'AORKEY'
  # - Enter '51777024' in the search bar and click on the 'search' button
  # - Select the one entry, and click on the 'Prepare Download' button
  # - Select Level2, Level1, Calibration, and Raw on the 'Download Options' pop up
  #   screen, and click on the 'Prepare Download' button

  # - When ready, download the data and unzip the file to your
  #   prefered location on your machine



Raw Light-curve Reduction
-------------------------

First of all, you will need some configuration files.  Copy the input
and configuration files for the demo from the `examples folder
<https://github.com/pcubillos/puppies/tree/master/examples/WASP43b_eclipse>`_
to your working directory.  You can find these files on your local
machine with the following Python script:

.. code-block:: python

    # This Python script shows you where the demo files are located:
    import puppies as p
    demo_folder = f'{p.ROOT}examples/WASP43b_eclipse/'
    print(demo_folder)

    # You can either copy the file from this folder to your current dir
    # using the command line or the following Python script:
    import os
    import shutil
    for file in os.listdir(demo_folder):
        shutil.copy(f'{demo_folder}/{file}', file)

.. warning:: **Text is TBD**

The same process can be executed from the Python Interpreter, after
importing the ``puppies`` package:

.. code-block:: python

  import pathlib
  import puppies as p

  root = str(pathlib.Path('.').absolute()) + '/'

  # New pup:
  cfile = root + 'pup_spitzer_WASP43b_eclipse.cfg'
  pup = p.init(cfile)

  # Run badpix:
  pup = p.io.load(root + "wa043b/" + "wa043b.p")
  p.core.badpix(pup)

  # Run centering:
  pup = p.io.load(root + "/wa043b/badpix/wa043b.p")
  cfile = root + "pup_center.cfg"
  p.core.center(pup, cfile)

  # Run photometry:
  pup = p.io.load(root + "/wa043b/badpix/gauss/wa043b.p")
  cfile = root + "pup_phot.cfg"
  p.core.photom(pup, cfile)



Light-curve Fitting and Retrieval
---------------------------------

.. warning:: **All of this is TBD**


The output vary depending on the selected run mode.  Additional low-
and mid-level routines are also available through this package.


Any of these steps can be run either interactively though the Python
Interpreter, or from the command line.  To streamline execution,
``puppies`` provides a single command to execute any of these runs.
To set up any of these runs, ``puppies`` uses configuration files
following the standard `INI format
<https://docs.python.org/3.6/library/configparser.html#supported-ini-file-structure>`_.

The :ref:`qexample` section above demonstrates a simple
secondary-eclipse analysis, while the next sections give a thorough
detail for each of the running modes.  Finally, most of the low- and
mid-level routines used for these calculations are available
through the ``puppies`` sub modules (see :ref:`API`).


------------------------------------------------------------------------

That's it, now let's see the results.  The screen outputs and any
warnings raised are saved into log files.  The output spectrum is
saved to a separate file, to see it, run this Python script (on
interactive mode, I suggest starting the session with ``ipython
--pylab``):

.. code-block:: python

  import matplotlib
  from scipy.ndimage.filters import gaussian_filter1d as gaussf
  import matplotlib.pyplot as plt
  plt.ion()
