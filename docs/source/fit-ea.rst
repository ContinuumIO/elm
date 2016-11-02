Fit EA
======

``elm`` can use an evolutionary algorithm for hyperparameterization.  This involves using the ``fit_ea`` method of ``elm.pipeline.Pipeline``.  It is helpful at this point to first read about ``elm.pipeline.Pipeline`` and how to configure a data source (TODO LINK pipeline.rst page somewhere) for the multi-model approaches in ``elm``.  That page summarizes how ``fit_ea`` and ``fit_ensemble`` may be fit to a single ``X`` matrix (when the keyword ``X`` is given) or a series of samples (when ``sampler`` and ``args_list`` are given).

The example below walks through configuring an evolutionary algorithm to select the best K-Means model with preprocessing steps inclusive of standard scaling and PCA.  First it sets up a sampler from HDF4 files (note the set up of a data source is the same as in ``fit_ensemble`` -TODO link to this same snippet there)

Example
-------
..code-block:: python

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

Next the example sets up a ``Pipeline`` of transformations

..code-block:: python

    def make_example_y_data(X, y=None, sample_weight=None, **kwargs):
        fitted = MiniBatchKMeans(n_clusters=5).fit(X.flat.values)
        y = fitted.predict(X.flat.values)
        return (X, y, sample_weight)

    pipeline_steps = [steps.Flatten(),
                         steps.ModifySample(make_example_y_data),
                         ('top_n', steps.SelectPercentile(percentile=80,score_func=f_classif)),
                         ('kmeans', MiniBatchKMeans(n_clusters=4))]
    pipeline = Pipeline(pipeline_steps, scoring=kmeans_aic)

The example above uses ``elm.pipeline.steps.ModifySample`` to return a ``y`` data set corresponding to ``X`` ``ElmStore`` so that the example can show ``SelectPercentile`` for feature selection.

Next ``evo_params`` need to be called by passing a ``param_grid`` dict to ``elm.model_selection.evolve.ea_setup``.  The ``param_grid`` uses scikit-learn syntax for parameter replacement (i.e. a named step like "kmeans" then a double underscore then a parameter name for that step ["n_clusters"]), so this ``param_grid`` could potentially run models with ``n_clusters`` in ``range(3, 10)`` and ``percentile`` in ``range(20, 100, 5)``. The ``control`` dict sets parameters for the evolutionary algorithm (described below - TODO LINK TO BELOW).

..code-block:: python

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

    with client_context() as client:
        fitted = pipeline.fit_ea(evo_params=evo_params,
                                 client=client,
                                 **data_source)
        preds = pipeline.predict_many(client=client, **data_source)

Reference ``param_grid`` - ``control``
--------------------------------------

TODO fill this out
 * **select_method**: selNSGA2
 * **crossover_method**: cxTwoPoint,
 * **mutate_method**: mutUniformInt,
 * **init_pop**: random,
 * **indpb**: 0.5,
 * **mutpb**: 0.9,
 * **cxpb**:  0.3,
 * **eta**:   20,
 * **ngen**:  2,
 * **mu**:    4,
 * **k**:     4,
 * **early_stop**: {'abs_change': [10], 'agg': 'all'},
# alternatively early_stop: {percent_change: [10], agg: all}
# alternatively early_stop: {threshold: [10], agg: any}

More Reading
------------

TODO Links to ``deap``