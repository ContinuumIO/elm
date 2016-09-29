
Environment variables used by elm
==========

The following are environment variables controlling how elm works:

 * `MAX_PARAM_RETRIES`: how many times to retry in genetic algorithm when parameters are repeatedly infeasible,
 * `DASK_EXECUTOR`: Dask executor to use. Choices `DISTRIBUTED, SERIAL, THREAD_POOL]` (default: `SERIAL`),
 * `DASK_SCHEDULER`: Dask scheduler url to use
 * `ELM_EXAMPLE_DATA_PATH`: Local clone dir of `ContinuumIO/elm-data` repo (useful in unit testing),
 * `ELM_TRAIN_PATH`: Directory in which to serialize trained models,
 * `ELM_TRANSFORM_PATH`: Directory in which to serialize trained transform models like PCA,
 * `ELM_PREDICT_PATH`: Directory for prediction outputs,
 * `ELM_LOGGING_LEVEL`: Logging level.  Choices: `INFO` (default) and `DEBUG`

Each of the environment variables above can also be specified as a command line argument to `elm-main` the console entry point for the elm machine learning pipeline.

The help below from `elm-main` shows how to specify these variables from the command line:

.. code-block:: bash 

    $ elm-main -h
    usage: elm-main [-h] [--config CONFIG] [--config-dir CONFIG_DIR]
                    [--dask-threads DASK_THREADS]
                    [--dask-processes DASK_PROCESSES]
                    [--max-param-retries MAX_PARAM_RETRIES]
                    [--dask-executor {DISTRIBUTED,SERIAL,THREAD_POOL}]
                    [--dask-scheduler DASK_SCHEDULER]
                    [--ladsweb-local-cache LADSWEB_LOCAL_CACHE]
                    [--hashed-args-cache HASHED_ARGS_CACHE]
                    [--elm-example-data-path ELM_EXAMPLE_DATA_PATH]
                    [--elm-train-path ELM_TRAIN_PATH]
                    [--elm-transform-path ELM_TRANSFORM_PATH]
                    [--elm-predict-path ELM_PREDICT_PATH]
                    [--elm-logging-level {INFO,DEBUG}]
                    [--elm-configs-path ELM_CONFIGS_PATH] [--echo-config]

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
