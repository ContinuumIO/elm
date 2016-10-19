NASA SBIR Phase I - Open Source Parallel Image Analysis and Machine Learning Pipeline
============


Using the Code
-----

To use this code:

- **Install:**

   Create the development environment:

.. code-block:: bash 

    conda env create

- **Activate the environment:**

.. code-block:: bash 

    source activate elm-env

(older versions of the code may have `elm` in place of `elm-env` above.  The environment name was changed to avoid conflict with `elm` package on anaconda.org.  The `elm-env` is uploaded to the nasasbir org on anaconda.org.)

Install the source:

.. code-block:: bash

    python setup.py develop


Clone the `elm-data` repo using Git LFS so that more tests can be run:

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
-----
.. code-block:: bash
    DASK_EXECUTOR=SERIAL LADSWEB_LOCAL_CACHE=`pwd` DASK_SCHEDULER=1 elm-download-ladsweb --config elm/config/defaults/defaults.yaml

(replacing the yaml if not using the default VIIRS Level 2 dataset)

Run the faster running tests:

.. code-block:: bash

    py.test -m "not slow"
    
.. code-block:: bash

    py.test

or get the verbose test output

.. code-block:: bash

    py.test -v

and cut and paste a test mark to run a specific test:

.. code-block:: bash

    py.test -k test_train_makes_args_kwargs_ok

Run the default pipeline yaml:
-----
In serial_:

.. code-block:: bash

    DASK_EXECUTOR=SERIAL elm-main --config elm/config/defaults/defaults.yaml  --echo-config

With dask-distributed_:

.. code-block:: bash

    dask-scheduler

In separate command prompts do this for each worker:

.. code-block:: bash

    dworker 10.0.0.10:8786 # or what dask-scheduler gave as IP

Then

.. code-block:: bash

    ELM_LOGGING_LEVEL=DEBUG DASK_EXECUTOR=DISTRIBUTED DASK_SCHEDULER=10.0.0.10:8786 elm-main --config elm/config/defaults/defaults.yaml  --echo-config

(You should modify the `ensembles` section of one of the configs in `elm/example_configs` to a larger ensemble to see a better parallel versus serial performance difference)

Config File Format
-----

It is easiest to copy the default config referenced above in snippets, and then follow [these instructions on editing the config](https://github.com/ContinuumIO/nasasbir/blob/master/README_config.md).


.. toctree::
   :titlesonly:
   :hidden:
   :maxdepth: 1

   Home <self>
   About <about>
   Getting started... <getting_started>
   Features <features>
   Building <build_package>
   Cluster Help <cluster_help>

.. _config-docs:

.. toctree::
   :maxdepth: 2

   api.rst
   Configuration <Config/index>
   Environment Variables <environment_vars>
   Feature Selection <feature_selection>
   Example Pipeline <sample_pipeline>
   Testing <testing>
   Training <train>
   Transforms <transform>
   Associated Projects <nasa_projects>
   FAQ <faq>
   Github source <http://github.com/ContinuumIO/elm>
