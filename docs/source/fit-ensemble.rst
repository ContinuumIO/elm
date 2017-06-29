Multi-Model Fitting I: Ensemble Fitting
=======================================

Ensemble fitting may be helpful in cases where the representative sample is large and/or model parameter or fitting uncertainty should be considered.

Ensemble fitting may:

 * Use one or more samples,
 * Use one or more models (:doc:`Pipeline<pipeline>` instances), and/or
 * Use one or more generations of fitting, with model selection logic on each generation

It is helpful to first read the section Data Sources for a :doc:`Pipeline<pipeline>` showing how to use either a single ``X`` matrix or a series of samples from a ``sampler`` callable.

Define a Sampler
----------------

.. _full script can be found here: https://github.com/ContinuumIO/elm/blob/master/examples/api_example.py

The example below uses a ``sampler`` function and ``args_list`` (list of unpackable args to ``sampler``) to fit to many samples.  The `full script can be found here`_.  First the script does some imports and sets up a ``sampler`` function that uses ``band_specs`` (see also :doc:`ElmStore<elm-store>`) to select a subset of bands in ``HDF4`` files.

.. code-block:: python

    import os
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import IncrementalPCA
    import numpy as np
    from elm.config.dask_settings import client_context
    from elm.model_selection.kmeans import kmeans_model_averaging, kmeans_aic
    from elm.pipeline import steps, Pipeline
    from earthio import *
    from elm.sample_util.band_selection import select_from_file
    from earthio.metadata_selection import example_meta_is_day

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

Define ``Pipeline`` Steps
-------------------------

.. _standard scaling: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

Next a :doc:`Pipeline<pipeline>` is configured that flattens the separate band rasters to a single 2-D ``DataArray`` (See also :ref:`transform-flatten`, uses `standard scaling`_ from scikit-learn, then transforms with ``IncrementalPCA`` with 2 ``partial_fit`` batches before K-Means.  The :doc:`Pipeline<pipeline>` constructor also takes a ``scoring`` callable and optional ``scoring_kwargs``.

.. code-block:: python

    pipeline_steps = [steps.Flatten(),
                      ('scaler', steps.StandardScaler()),
                      ('pca', steps.Transform(IncrementalPCA(n_components=4), partial_fit_batches=2)),
                      ('kmeans', MiniBatchKMeans(n_clusters=4, compute_labels=True)),]
    pipe = Pipeline(pipeline_steps, scoring=kmeans_aic, scoring_kwargs=dict(score_weights=[-1]))

.. _signature for kmeans_aic: https://github.com/ContinuumIO/elm/blob/master/elm/model_selection/kmeans.py


See the `signature for kmeans_aic`_ here to write a similar scoring function, otherwise ``scoring`` defaults to calling the estimator's ``.score`` callable or exception if ``.score`` is not defined.

Configure Ensemble
------------------

Now we can call ``fit_ensemble`` after choosing some controls on the size of the ensemble, the number of generations, and the logic for selecting models after each generation.

Here's an example:

.. code-block:: python

    ensemble_kwargs = {
        'model_selection': kmeans_model_averaging,
        'model_selection_kwargs': {
            'drop_n': 2,
            'evolve_n': 2,
        },
        'init_ensemble_size': 4,
        'ngen': 3,
        'partial_fit_batches': 2,
        'saved_ensemble_size': 4,
        'models_share_sample': True,
    }

In the example above:

 * ``ngen`` sets the number of generations to 3
 * There are 4 initial ensemble members (``init_ensemble_size``),
 * After each generation ``kmeans_model_averaging`` (See :doc:`API docs<api>`) is called on the ensemble with ``model_selection_kwargs`` are passed.
 * There are 3 ``partial_fit`` batches for ``MiniBatchKMeans`` on every :doc:`Pipeline<pipeline>` instance (``partial_fit`` within the ``IncrementalPCA`` was configured in the initialization of ``steps.Transform`` above)
 * ``models_share_sample`` is set to ``True`` so in each generation every ensemble member is fit to the same sample, then on the next generation, every model is fit to the next sample determined by ``sampler`` and ``args_list`` in this case.  If ``models_share_sample`` were ``False``, then in each generation every ensemble member would be copied and fit to every sample, repeating the process on each generation.

.. _dask-distributed: https://distributed.readthedocs.io/en/latest/quickstart.html#setup-dask-distributed-the-hard-way

Fitting with Dask-Distributed
-----------------------------

In the snippets above, we have a ``data_source`` ``dict`` with ``sampler``,``band_specs`` and ``args_list`` key / values.  We can pass this with the ``ensemble_kwargs`` ensemble configuration to ``fit_ensemble`` as well as :doc:`predict_many<predict-many>` . The data source for :doc:`predict_many<predict-many>` does not necessarily have to be the same one given to ``fit_ensemble`` or ``fit_ea``).

**Note** : If you want ``dask-distributed`` as a client, first make sure you are running a ``dask-scheduler`` and ``dask-worker`` .  Read more here on `dask-distributed`_ and follow instructions in :doc:`environment variables<environment-vars>` .

.. code-block:: python

    with client_context() as client:
        ensemble_kwargs['client'] = client
        pipe.fit_ensemble(**data_source, **ensemble_kwargs)
        pred = pipe.predict_many(client=client, **data_source)

Fitting with ``dask`` parallelizes over the ensemble members (:doc:`Pipeline<pipeline>` instances) and over the calls to ``partial_fit``  - currently transformers in the ``Pipeline`` are not parallelized with ``dask`` .

.. _controlling-ensemble:

Controlling Ensemble Initialization
-----------------------------------

To initialize the ensemble with :doc:`Pipeline<pipeline>` instances that do not all share the same parameters (as above), we could replace ``init_ensemble_size`` above with ``ensemble_init_func``

.. code-block:: python

    n_clusters_choices = tuple(range(4, 9))
    def ensemble_init_func(pipe, **kwargs):
        models = []
        for c in n_clusters_choices:
            new_pipe = pipe.new_with_params(kmeans__n_clusters=c)
            models.append(new_pipe)
        return models
    ensemble_kwargs = {
        'model_selection': kmeans_model_averaging,
        'model_selection_kwargs': {
            'drop_n': 2,
            'evolve_n': 2,
        },
        'ensemble_init_func': ensemble_init_func,
        'ngen': 3,
        'partial_fit_batches': 2,
        'saved_ensemble_size': 4,
        'models_share_sample': True,
    }
    with client_context() as client:
        ensemble_kwargs['client'] = client
        pipe.fit_ensemble(**data_source, **ensemble_kwargs)
        pred = pipe.predict_many(client=client, **data_source)

In the example above, ``Pipeline.new_with_params(kmeans__n_clusters)`` uses the scikit-learn syntax for parameter modifications of named steps in a pipeline.  In the initialization of :doc:`Pipeline<pipeline>` in the example above, the ``MiniBatchMeans`` step was named ``kmeans``, so ``kmeans__n_clusters=c`` sets the ``n_clusters`` parameter to the K-Means step and the ensemble in this case consists of one :doc:`Pipeline<pipeline>` for each of ``n_clusters`` choices in `(4, 5, 6, 7, 8)`.

