Environment Variables
=====================

The following are environment variables control ``elm-main`` and are also inputs to other ``elm`` functions like ``elm.config.client_context`` (a dask client context):

 * ``DASK_EXECUTOR``: Dask executor to use. Choices ``[DISTRIBUTED, SERIAL, THREAD_POOL]`` (default: ``SERIAL``)
 * ``DASK_SCHEDULER``: Dask scheduler URL, such as ``10.0.0.10:8786``, if using ``DASK_EXECUTOR=DISTRIBUTED``
 * ``DASK_THREADS``: Number of threads if using ``DASK_EXECUTOR==THREAD_POOL``
 * ``ELM_EXAMPLE_DATA_PATH``: Path to local clone of http://github.com/ContinuumIO/elm-examples (used for ``py.test``)
 * ``ELM_LOGGING_LEVEL``: Either ``INFO`` (default) or ``DEBUG``
 * ``ELM_PREDICT_PATH``: Base path for saving prediction output
 * ``ELM_TRAIN_PATH``: Base path for saving trained ensembles
 * ``MAX_PARAM_RETRIES``: How many times to retry in genetic algorithm when parameters are repeatedly infeasible
 * ``IS_TRAVIS``:  If ``IS_TRAVIS=1`` , then dask's distributed client is not used (the client if started in CI tests can cause hanging)

