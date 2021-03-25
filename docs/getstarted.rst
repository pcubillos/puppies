.. _getstarted:

Getting Started
===============

``puppies`` is a general Python library to analyze exoplanet time-series observations (you could say, a general TSO type of tool).

System Requirements
-------------------

``puppies`` is compatible with Python3.6+ and has been `tested <https://travis-ci.com/pcubillos/puppies>`_ to work in both Linux and OS X, with the following software:

* numpy >= 1.8.1
* scipy >= 0.13.3
* matplotlib >= 1.3.1
* astropy >= 3.1
* mc3 >= 3.0.6

.. * sphinx (version 1.7.9+)
   * sphinx_rtd_theme (version 0.4.2+)
   * packaging (version 17.1+)


.. _install:

Install
-------

To install ``puppies`` run the following command from the terminal:

.. code-block:: shell

    pip install exo_puppies

Or if you prefer conda:

.. code-block:: shell

    conda install -c conda-forge puppies

Alternatively (e.g., for developers), clone the repository to your local machine with the following terminal commands:

.. code-block:: shell

  git clone https://github.com/pcubillos/puppies
  cd puppies
  python setup.py develop

------------------------------------------------------------

Once installed, take a look at the ``puppies`` main menu by executing the following command:

.. code-block:: shell

  # Display puppies main help menu:
  pup -h

From there, take a look at the sub-command helps or the rest of these docs for further details, or see the :ref:`qexample` for an introductory worked example.


.. _qexample:

Quick Example
-------------

What's a good quick example?

.. code-block:: shell

  # Show the dog of the day on the browser:
  pup --day


