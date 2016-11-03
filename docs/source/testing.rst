Testing ``elm``
----------
This page describes how to run the ``py.test`` unit tests and run all or some scripts and configs in ``elm-examples`` - TODO LINK.


``py.test`` Unit Tests
~~~~~~~~~~~~~~~~~~~~~~

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

When running ``py.test`` the environment variables related to dask determine whether dask-distributed or thread pool client or serial evaluation is used- TODO LINK are

Longer Running Tests
~~~~~~~~~~~~~~~~~~~~

The ``elm-run-all-tests`` console entry point can automate running of some or all python scripts and ``yaml`` ``elm-main`` config files in ``elm-examples`` and/or the ``py.test`` unit tests.

Here is an example that is run from inside the cloned ``elm`` repository with ``elm-examples`` cloned in the current directory (see the first two arguments: `./` - cloned ``elm`` repo and ``./elm-examples`` - the location of cloned ``elm-examples``)

.. code-block:: bash


   ELM_LOGGING_LEVEL=DEBUG elm-run-all-tests ./ elm-examples/ --skip-pytest --skip-scripts --dask-clients SERIAL DISTRIBUTED --dask-scheduler 10.0.0.10:8786

Above the arguments ``--skip-scripts`` and ``skip-pytest`` refer to skipping the scripts in ``elm-examples`` and ``py.test``s in ``elm``, respectively, so the command above will run the all configs in ``elm-examples/configs`` once for serial evaluation and once with dask-distributed.

Here is the full help on ``elm-run-all-tests`` entry point:

.. code-block:: bash

    $ elm-run-all-tests --help
    usage: elm-run-all-tests [-h] [--pytest-mark PYTEST_MARK]
                             [--dask-clients {ALL,SERIAL,DISTRIBUTED,THREAD_POOL} [{ALL,SERIAL,DISTRIBUTED,THREAD_POOL} ...]]
                             [--dask-scheduler DASK_SCHEDULER] [--skip-pytest]
                             [--skip-scripts] [--skip-configs]
                             [--add-large-test-settings]
                             [--glob-pattern GLOB_PATTERN]
                             [--remote-git-branch REMOTE_GIT_BRANCH]
                             repo_dir elm_examples_path

    Run longer-running tests of elm

    positional arguments:
      repo_dir              Directory that is the top dir of cloned elm repo
      elm_examples_path     Path to a directory which contains subdirectories
                            "scripts", "scripts", and "example_data" with yaml-
                            configs, python-scripts, and example data,
                            respectively

    optional arguments:
      -h, --help            show this help message and exit
      --pytest-mark PYTEST_MARK
                            Mark to pass to py.test -m (marker of unit tests)
      --dask-clients {ALL,SERIAL,DISTRIBUTED,THREAD_POOL} [{ALL,SERIAL,DISTRIBUTED,THREAD_POOL} ...]
                            Dask client(s) to test: ['ALL', 'SERIAL',
                            'DISTRIBUTED', 'THREAD_POOL']
      --dask-scheduler DASK_SCHEDULER
                            Dask scheduler URL
      --skip-pytest         Do not run py.test (default is run py.test as well as
                            configs)
      --skip-scripts        Do not run scripts from elm-examples
      --skip-configs        Do not run configs from elm-examples
      --add-large-test-settings
                            Adjust configs for larger ensembles / param_grids
      --glob-pattern GLOB_PATTERN
                            Glob within repo_dir
      --remote-git-branch REMOTE_GIT_BRANCH
                            Run on a remote git branch


