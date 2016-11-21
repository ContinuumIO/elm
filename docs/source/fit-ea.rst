Fit EA
======

``elm`` can use an evolutionary algorithm for hyperparameterization.  This involves using the ``fit_ea`` method of :doc:`Pipeline<pipeline>`.  It is helpful at this point to first read about :doc:`Pipeline<pipeline>` and how to configure a data source for the multi-model approaches in ``elm``.  That page summarizes how :doc:`fit_ea<fit-ea>` and :doc:`fit_ensemble<fit-ensemble>` may be fit to a single ``X`` matrix (when the keyword ``X`` is given) or a series of samples (when ``sampler`` and ``args_list`` are given).

The example below walks through configuring an evolutionary algorithm to select the best K-Means model with preprocessing steps inclusive of standard scaling and PCA.  First it sets up a sampler from HDF4 files (note the set up of a data source is the same as in :doc:`fit_ensemble<fit-ensemble>`)

Example
-------
.. code-block:: python

    import os

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_selection import SelectPercentile, f_classif
    import numpy as np

    from elm.config.dask_settings import client_context
    from elm.model_selection.evolve import ea_setup
    from elm.model_selection.kmeans import kmeans_model_averaging, kmeans_aic
    from elm.pipeline import Pipeline, steps
    from elm.readers import *
    from elm.sample_util.band_selection import select_from_file
    from elm.sample_util.metadata_selection import example_meta_is_day

    ELM_EXAMPLE_DATA_PATH = os.environ['ELM_EXAMPLE_DATA_PATH']

    band_specs = list(map(lambda x: BandSpec(**x),
            [{'search_key': 'long_name', 'search_value': "Band 1 ", 'name': 'band_1'},
             {'search_key': 'long_name', 'search_value': "Band 2 ", 'name': 'band_2'},
             {'search_key': 'long_name', 'search_value': "Band 3 ", 'name': 'band_3'},
             {'search_key': 'long_name', 'search_value': "Band 4 ", 'name': 'band_4'},
             {'search_key': 'long_name', 'search_value': "Band 5 ", 'name': 'band_5'},
             {'search_key': 'long_name', 'search_value': "Band 6 ", 'name': 'band_6'},
             {'search_key': 'long_name', 'search_value': "Band 7 ", 'name': 'band_7'}]))
    # Just get daytime files
    HDF4_FILES = [f for f in glob.glob(os.path.join(ELM_EXAMPLE_DATA_PATH, 'hdf4', '*hdf'))
                  if example_meta_is_day(load_hdf4_meta(f))]
    data_source = {
        'sampler': select_from_file,
        'band_specs': band_specs,
        'args_list': HDF4_FILES,
    }

Next the example sets up a :doc:`Pipeline<pipeline>` of transformations

.. code-block:: python

    def make_example_y_data(X, y=None, sample_weight=None, **kwargs):
        fitted = MiniBatchKMeans(n_clusters=5).fit(X.flat.values)
        y = fitted.predict(X.flat.values)
        return (X, y, sample_weight)

    pipeline_steps = [steps.Flatten(),
                      steps.ModifySample(make_example_y_data),
                      ('top_n', steps.SelectPercentile(percentile=80,score_func=f_classif)),
                      ('kmeans', MiniBatchKMeans(n_clusters=4))]
    pipeline = Pipeline(pipeline_steps, scoring=kmeans_aic, scoring_kwargs=dict(score_weights=[-1]))

The example above uses ``elm.pipeline.steps.ModifySample`` to return a ``y`` data set corresponding to ``X`` ``ElmStore`` so that the example can show ``SelectPercentile`` for feature selection.

Next ``evo_params`` need to be called by passing a ``param_grid`` dict to ``elm.model_selection.evolve.ea_setup``.  The ``param_grid`` uses scikit-learn syntax for parameter replacement (i.e. a named step like "kmeans" then a double underscore then a parameter name for that step ["n_clusters"]), so this ``param_grid`` could potentially run models with ``n_clusters`` in ``range(3, 10)`` and ``percentile`` in ``range(20, 100, 5)``. The ``control`` dict sets parameters for the evolutionary algorithm (described below).

.. code-block:: python

    param_grid =  {
        'kmeans__n_clusters': list(range(3, 10)),
        'top_n__percentile': list(range(20, 100, 5)),
        'control': {
            'select_method': 'selNSGA2',
            'crossover_method': 'cxTwoPoint',
            'mutate_method': 'mutUniformInt',
            'init_pop': 'random',
            'indpb': 0.5,
            'mutpb': 0.9,
            'cxpb':  0.3,
            'eta':   20,
            'ngen':  2,
            'mu':    4,
            'k':     4,
            'early_stop': {'abs_change': [10], 'agg': 'all'},
            # alternatively early_stop: {percent_change: [10], agg: all}
            # alternatively early_stop: {threshold: [10], agg: any}
        }
    }

    evo_params = ea_setup(param_grid=param_grid,
                          param_grid_name='param_grid_example',
                          score_weights=[-1]) # minimization

.. _dask-distributed: https://distributed.readthedocs.io/en/latest/quickstart.html#setup-dask-distributed-the-hard-way

Running with ``dask`` to parallelize over the individual solutions (:doc:`Pipeline<pipeline>` instances) and their calls to ``partial_fit`` .

**Note** : If you want ``dask-distributed`` as a client, first make sure you are running a ``dask-scheduler`` and ``dask-worker`` .  Read more here on `dask-distributed`_ and follow instructions in :doc:`environment variables<environment-vars>` .

.. code-block:: python

    with client_context() as client:
        fitted = pipeline.fit_ea(evo_params=evo_params,
                                 client=client,
                                 **data_source)
        preds = pipeline.predict_many(client=client, **data_source)

Reference ``param_grid`` - ``control``
--------------------------------------

In the example above the ``param_grid`` has a ``control`` dictionary specifying parameters of the evolutionary algorithm.  The ``control`` dict names the functions to be used for crossover, mutation, and selection, and the other arguments are passed to the those methods as needed.  The following section describes each key/value of a ``control`` dictionary.

**Note** While it is possible to change the ``select_method``, ``crossover_method`` and ``mutate_method`` below from the example shown, it is important to use methods that are consistent with how ``fit_ea`` expresses parameter choices.  For each parameter in the ``param_grid``, such as ``kmeans__n_clusters=list(range(3, 10))``, ``fit_ea`` optimizes with *indices* into ``kmeans__n_clusters`` list, i.e. choosing among ``list(range(7))``, *not* optimizing an integer parameter between 3 and 10.  This allows ``fit_ea`` to avoid custom treatment of string, float, or integer data types in the parameters' lists of choices.  If changing the ``mutate_method`` keep in mind that it needs to take individuals that are sequences of integers as arguments and return the same.

.. _see the list of selection methods here: http://deap.gel.ulaval.ca/doc/dev/api/tools.html#selection

.. _crossover method from deap.tools: http://deap.gel.ulaval.ca/doc/dev/api/tools.html#crossover

.. _mutation methods: http://deap.gel.ulaval.ca/doc/dev/api/tools.html#mutation

 * **select_method**: Selection method on each generation of evolutionary algorithm.  The selection method is typically ``selNSGA2`` but can be any ``deap.tools`` selection method (see the `list of selection methods here`_)
 * **crossover_method**: Crossover method between two individuals, e.g. ``cxTwoPoint``, or any `crossover method from deap.tools`_
 * **mutate_method**: Mutation method, typically ``mutUniformInt``, or another mutation method from ``deap.tools`` `mutation methods`_
 * **init_pop**: Placeholder for initialization features- must always be ``random`` (random initialization of solutions)
 * **indpb**: Proability each attribute (feature) is mutated when an individual is mutated, e.g. ``0.5`` (passed to mutation methods in ``deap.tools``)
 * **mutpb**: When two individuals crossover, this is the probability they will mutate immediately after crossover, e.g. ``0.9``
 * **cxpb**:  Probabity of crossover ``0.3``
 * **eta**:   Tuning parameter in NSGA-2 - passed to mutate and mate methods.  With a higher ``eta`` crowding is penalized and offspring are more different from their parents
 * **ngen**:  Number of generations in genetic algorithm
 * **mu**: Size of the population of solutions (individuals) initially
 * **k**: Select the top ``k`` on each generation
 * **early_stop**: Control stopping of algorithm before ``ngen`` number of generations is completed.  Examples are below (note ``agg`` refers to aggregation as ``all`` or ``any`` in the case it is a multi-objective problem)

   * *Stop on absolute change in objective*: ``{'abs_change': [10], 'agg': 'all'}``
   * *Stop on percent change in objective*: ``early_stop: {percent_change: [10], agg: all}``
   * *Stop on reaching objective threshold*: ``early_stop: {threshold: [10], agg: any}``

More Reading
------------

.. _deap Docs: http://deap.readthedocs.io/en/master/

.. _deap source code: https://github.com/deap

.. _deap NSGA-2 example on which fit_ea is based: https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py

``fit_ea`` relies on ``deap`` for Pareto sorting and the genetic algorithm components described above.  Read more about ``deap``:

 * `deap Docs`_
 * `deap source code`_
 * `deap NSGA-2 example on which fit_ea is based`_

