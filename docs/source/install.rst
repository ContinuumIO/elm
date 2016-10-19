Install ELM
===========

You can install elm with ``conda`` or by installing from source.

Install from Conda
~~~~~~

To install the latest release of elm

.. code-block:: bash

    conda install elm -c elm

This installs elm and all common dependencies.


Install from Source
~~~~~~

To install elm from source, clone the repository from `github
<https://github.com/ContinuumIO/elm>`_:

.. code-block:: bash

    git clone https://github.com/ContinuumIO/elm.git
    cd elm
    python setup.py install


**Create the development environment:**

.. code-block:: bash

    conda env create

**Activate the environment:**

.. code-block:: bash

    source activate elm-env

(older versions of the code may have ``elm`` in place of ``elm-env`` above.  The environment name was changed to avoid conflict with ``elm`` package on anaconda.org.  The ``elm-env`` is uploaded to the nasasbir org on anaconda.org.)

Install the source:

.. code-block:: bash

    python setup.py develop


Clone the ``elm-data`` repo using Git LFS so that more tests can be run:

.. code-block:: bash

    brew install git-lfs # or apt-get, yum, etc
    git lfs install
    git clone https://github.com/ContinuumIO/elm-data
    git remote add origin https://github.com/ContinuumIO/elm-data

Add the following to your .bashrc or environment, changing the paths depending on where you have cloned elm-data:
.. code-block:: bash
    export DASK_EXECUTOR=SERIAL
    export ELM_EXAMPLE_DATA_PATH=/Users/psteinberg/Documents/elm-data


Run the default config
~~~~~~
.. code-block:: bash
    DASK_EXECUTOR=SERIAL LADSWEB_LOCAL_CACHE=`pwd` DASK_SCHEDULER=1 elm-download-ladsweb --config elm/config/defaults/defaults.yaml

(replacing the yaml if not using the default VIIRS Level 2 dataset)

Run the faster running tests:

.. code-block:: bash

    py.test -m "not slow"

.. code-block:: bash

    py.test


Test
~~~~~~

Test elm with ``py.test``::

.. code-block:: bash

    py.test -m "not slow"


or get the verbose test output

.. code-block:: bash

    py.test -v

and cut and paste a test mark to run a specific test:

.. code-block:: bash

    py.test -k test_train_makes_args_kwargs_ok
