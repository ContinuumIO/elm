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
    conda env create
    source activate elm-env
    python setup.py develop

Clone the ``elm-data`` repo using Git LFS so that more tests can be run:

.. code-block:: bash

    brew install git-lfs # or apt-get, yum, etc
    git lfs install
    git clone https://github.com/ContinuumIO/elm-data
    git remote add origin https://github.com/ContinuumIO/elm-data

Add the following to your .bashrc or environment, changing the paths depending on where you have cloned elm-data:

.. code-block:: bash

    export ELM_EXAMPLE_DATA_PATH=/Users/peter/Documents/elm-data

Run one of the example configs:

.. code-block:: bash

    elm-main --config elm-examples/configs/kmeans_hdf4.yaml

Or run all of the example configs:

.. code-block:: bash

    elm-main --config-dir elm-examples/configs/

If you have dask-distributed scheduler and worker(s) running or want to run with a dask ``ThreadPool``, ``elm-main`` can be called with environment variables (described here - TODO LINK) to use dask-distributed

.. code-block:: bash

    export DASK_EXECUTOR=DISTRIBUTED
    export DASK_SCHEDULER=10.0.0.10:8786
    elm-main --config elm-examples/configs/kmeans_hdf4.yaml

Test
~~~~~~

These testing instructions assume you have cloned the ``elm`` repository locally and installed from source - TODO LINK

*Note:* Many tests are skipped if you have not defined the environment variable ``ELM_EXAMPLE_DATA_PATH`` (referring to your local clone of http://github.com/ContinuumIO/elm-examples)

Run the faster running tests:

.. code-block:: bash

    py.test -m "not slow"

Running all tests:

.. code-block:: bash

    py.test

or get the verbose test output

.. code-block:: bash

    py.test -v

and cut and paste a test mark to run a specific test:

.. code-block:: bash

    py.test -k test_bad_train_config
