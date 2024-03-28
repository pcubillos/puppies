.. _install:

Install
-------

To install ``puppies`` run the following command from the terminal:

.. code-block:: shell

    pip install exo_puppies

Or if you prefer conda:

.. code-block:: shell

    conda install -c conda-forge exo_puppies

Alternatively (e.g., for developers), clone and install the repository
to your local machine with the following terminal commands:

.. code-block:: shell

  git clone https://github.com/pcubillos/puppies
  cd puppies
  pip install -r requirements.txt
  pip install -e .


``puppies`` is compatible with Python3.7+ and has been `tested
<https://github.com/pcubillos/bibmanager/actions/workflows/python-package.yml>`_
to work in both Linux and OS X.

Now you are ready for some *puppies*!
  
------------------------------------------------------------


Dog of the Day
--------------

You can display the dog of the day on your browser with the following
terminal command:

.. code-block:: shell

  # Show the dog of the day on the browser:
  pup --day
