``elm-main`` Entry Point
========================

``elm-main`` runs the training and prediction steps configured in a ``yaml`` file, as described in the ``elm`` ``yaml`` Specs section - TODO LINK.

The simple use of ``elm-main`` is to run one ``yaml`` config:

.. code-block:: bash

   elm-main --config elm-examples/configs/kmeans_hdf4.yaml

``elm-main`` uses the :doc:`environment variables described here<environment-vars>`, many of which may be overriden by optional arguments to ``elm-main`` (see below).

The next section goes over all the command line options for ``elm-main``

Config(s) To Run
----------------

``elm-main`` can run a single ``yaml`` config or a directory of ``yaml`` config files. To run with a single ``yaml`` file, use the ``--config`` argument as above, or to run with a directory of config ``yaml`` files, use ``--config-dir``

Controlling Train vs. Predict
-----------------------------

The following arguments control which parts of the config are being run:

 * ``--train-only``: Run only the training actions specified in the ``run`` section of config
 * ``--predict-only``: Run only the predict actions specified in config

Overriding Arguments to ``fit_ensemble``
----------------------------------------

The following arguments, if given, will override similarly named values in the ``yaml`` config that are associated with the ``train`` section (configuration of a final estimator in an :doc:`Pipeline<pipeline>`:

 * ``--partial-fit-batches``: Number of ``partial_fit`` batches for final estimator (this does not control ``partial_fit`` batches within a ``transform`` step in ``run`` - ``pipeline`` steps)
 * ``--init-ensemble-size``: Initial ensemble size (ignored if ``ensemble_init_func`` is given in config(s))
 * ``--saved-ensemble-size``: Final number of trained models to serialize in each ``train`` action of the ``run`` section

Use Dask Client
---------------

To use a ``dask-distributed`` or dask ``ThreadPool`` client, use the :doc:`environment variables described here<environment-vars>` - or override them with command line arguments to ``elm-main``:

 * ``--dask-executor``: One of \[``DISTRIBUTED``  ``SERIAL`` or ``THREAD_POOL`` \]
 * ``--dask-scheduler``: Dask-distributed scheduler url, e.g. ``10.0.0.10:8786``

Directories for Serialization
-----------------------------

The following arguments control where trained models and predictions are saved:

 * ``--elm-train-path``: Trained ``Pipeline`` instances are saved here - see also ``ELM_TRAIN_PATH`` environment variable - TODO LINK
 * ``--elm-predict-path``: Predictions are saved here - see also ``ELM_PREDICT_PATH`` environment variable

Help for ``elm-main``
--------------------------

Here is the full help for ``elm-main``:

.. code-block:: bash

    $ elm-main --help
    usage: elm-main [-h] [--config CONFIG | --config-dir CONFIG_DIR]
                    [--train-only | --predict-only]
                    [--partial-fit-batches PARTIAL_FIT_BATCHES]
                    [--init-ensemble-size INIT_ENSEMBLE_SIZE]
                    [--saved-ensemble-size SAVED_ENSEMBLE_SIZE] [--ngen NGEN]
                    [--dask-threads DASK_THREADS]
                    [--max-param-retries MAX_PARAM_RETRIES]
                    [--dask-executor {DISTRIBUTED,SERIAL,THREAD_POOL}]
                    [--dask-scheduler DASK_SCHEDULER]
                    [--elm-example-data-path ELM_EXAMPLE_DATA_PATH]
                    [--elm-train-path ELM_TRAIN_PATH]
                    [--elm-predict-path ELM_PREDICT_PATH]
                    [--elm-logging-level {INFO,DEBUG}]

    Pipeline classifier / predictor using ensemble and partial_fit methods

    optional arguments:
      -h, --help            show this help message and exit
      --train-only          Run only the training, not prediction, actions
                            specified by config
      --predict-only        Run only the prediction, not training, actions
                            specified by config
      --echo-config         Output running config as it is parsed

    Inputs:
      Input config file or directory

      --config CONFIG       Path to yaml config
      --config-dir CONFIG_DIR
                            Path to a directory of yaml configs

    Run:
      Run options

    Control:
      Keyword arguments to elm.pipeline.ensemble

      --partial-fit-batches PARTIAL_FIT_BATCHES
                            Partial fit batches (for estimator specified in
                            config's "train"
      --init-ensemble-size INIT_ENSEMBLE_SIZE
                            Initial ensemble size (ignored if using
                            "ensemble_init_func"
      --saved-ensemble-size SAVED_ENSEMBLE_SIZE
                            How many of the "best" models to serialize
      --ngen NGEN           Number of ensemble generations, defaulting to ngen
                            from ensemble_kwargs in config

    Environment:
      Compute settings (see also help on environment variables)

      --dask-threads DASK_THREADS
                            See also env var DASK_THREADS
      --dask-processes DASK_PROCESSES
                            See also env var DASK_PROCESSES
      --max-param-retries MAX_PARAM_RETRIES
                            See also env var MAX_PARAM_RETRIES
      --dask-executor {DISTRIBUTED,SERIAL,THREAD_POOL}
                            See also DASK_EXECUTOR
      --dask-scheduler DASK_SCHEDULER
                            See also DASK_SCHEDULER
      --elm-example-data-path ELM_EXAMPLE_DATA_PATH
                            See also ELM_EXAMPLE_DATA_PATH
      --elm-train-path ELM_TRAIN_PATH
                            See also ELM_TRAIN_PATH
      --elm-predict-path ELM_PREDICT_PATH
                            See also ELM_PREDICT_PATH
      --elm-logging-level {INFO,DEBUG}
                            See also ELM_LOGGING_LEVEL
      --elm-configs-path ELM_CONFIGS_PATH
                            See also ELM_CONFIGS_PATH
      --elm-large-test ELM_LARGE_TEST
                            See also ELM_LARGE_TEST
    Pipeline classifier / predictor using ensemble and partial_fit methods

    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Path to yaml config
      --config-dir CONFIG_DIR
                            Path to a directory of yaml configs
      --dask-threads DASK_THREADS
                            See also env var DASK_THREADS
      --dask-processes DASK_PROCESSES
                            See also env var DASK_PROCESSES
      --max-param-retries MAX_PARAM_RETRIES
                            See also env var MAX_PARAM_RETRIES
      --dask-executor {DISTRIBUTED,SERIAL,THREAD_POOL}
                            See also DASK_EXECUTOR
      --dask-scheduler DASK_SCHEDULER
                            See also DASK_SCHEDULER
      --ladsweb-local-cache LADSWEB_LOCAL_CACHE
                            See also LADSWEB_LOCAL_CACHE
      --hashed-args-cache HASHED_ARGS_CACHE
                            See also HASHED_ARGS_CACHE
      --elm-example-data-path ELM_EXAMPLE_DATA_PATH
                            See also ELM_EXAMPLE_DATA_PATH
      --elm-train-path ELM_TRAIN_PATH
                            See also ELM_TRAIN_PATH
      --elm-transform-path ELM_TRANSFORM_PATH
                            See also ELM_TRANSFORM_PATH
      --elm-predict-path ELM_PREDICT_PATH
                            See also ELM_PREDICT_PATH
      --elm-logging-level {INFO,DEBUG}
                            See also ELM_LOGGING_LEVEL
      --elm-configs-path ELM_CONFIGS_PATH
                            See also ELM_CONFIGS_PATH
      --echo-config         Output running config as it is parsed
