Pipeline
========

Overview of ``Pipeline`` in ``elm``
-----------------------------------

``elm.pipeline.Pipeline`` allows a sequence of transformations on samples before fitting, transforming, and/or predicting from an scikit-learn estimator.  ``elm.pipeline.Pipeline`` is similar to the concept of the ``Pipeline`` in scikit-learn (``sklearn.pipeline.Pipeline``) but differs in several ways described below.

.. _xarray.Dataset: http://xarray.pydata.org/en/stable/api.html#dataset

* **Data sources for a Pipeline**: In ``elm``, the fitting expects ``X`` to be an :doc:`ElmStore<elm-store>` or `xarray.Dataset`_ rather than a ``numpy`` array as in scikit-learn.  This allows the ``Pipeline`` of transformations to include operations on cubes and other data structures common in satellite data machine learning.
* **Transformations**: In scikit-learn each step in a ``Pipeline`` passes a numpy array to the next step by way of a ``fit_transform`` method.  In ``elm``, a ``Pipeline`` always passes a tuple of `(X, y, sample_weight)` where `X` is an :doc:`ElmStore<elm-store>` or ``xarray.Dataset`` and ``y`` and ``sample_weight`` are numpy arrays or ``None``.
* **Partial Fit for Large Samples**: In ``elm`` a transformer with a ``partial_fit`` method, such as ``sklearn.decomposition.IncrementalPCA`` may be partially fit several times as a step in a ``Pipeline`` and the final estimator may also use ``partial_fit`` several times with ``dask-distributed`` for parallelization.
* **Multi-Model / Multi-Sample Fitting**: In ``elm``, a ``Pipeline`` can be fit with:
   * :doc:`fit_ensemble<fit-ensemble>`: This method repeats model fitting over a series of samples and/or a ensemble of ``Pipeline`` instances.  The ``Pipeline`` instances in the ensemble may or may not have the same initialization parameters.  :doc:`fit_ensemble<fit-ensemble>` can run in generations, optionally applying user-given model selection logic between generations.  This :doc:`fit_ensemble<fit-ensemble>` method is aimed at improved model fitting in cases where a representative sample is large and/or there is a need to account for parameter uncertainty.
   * :doc:`fit_ea<fit-ea>`:  This method uses `Distributed Evolutionary Algorithms in Python`_ (``deap``) to run a genetic algorithm, typically NSGA-2, that selects the best ``Pipeline`` instance(s).  The interface for :doc:`fit_ea<fit-ea>` and :doc:`fit_ensemble<fit-ensemble>` are similar, but :doc:`fit_ea<fit-ea>` takes an ``evo_params`` argument to configure the genetic algorithm.
* **Multi-Model / Multi-Sample Prediction**: ``elm``'s ``Pipeline`` has a method :doc:`predict_many<predict-many>` that can use dask-distributed to predict from one or more ``Pipeline`` instances and/or one or more samples (:doc:`ElmStore<elm-store>`s or ``xarray.Datasets``s).  By default :doc:`predict_many<predict-many>` will predict for all models in the final ensemble output by :doc:`fit_ensemble<fit-ensemble>`.

.. _Distributed Evolutionary Algorithms in Python: http://deap.readthedocs.io/en/master/

The following discusses each step of making a ``Pipeline`` that uses most of the features described above.

Data Sources for a ``Pipeline``
-----------------------------------
``Pipeline`` can be used for fitting / transforming / predicting from a single sample or series of samples.  For the :doc:`fit_ensemble<fit-ensemble>`, :doc:`fit_ea<fit-ea>` or :doc:`predict_many<predict-many>` methods of a ``Pipeline`` instance:
 * To fit to a single sample, use the ``X`` keyword argument, and optionally ``y`` and ``sample_weight`` keyword arguments.
 * To fit to a series of samples, use the ``args_list`` and ``sampler`` keyword arguments.

If ``X`` is given it is assumed to be an :doc:`ElmStore<elm-store>` or `xarray.Dataset``

If ``sampler`` is given with ``args_list``, then each element of ``args_list`` is unpacked as arguments to the callable ``sampler``.  There is a special case of giving ``sampler`` as ``elm.readers.band_selection.select_from_file`` which allows using the functions from ``elm.readers`` for reading common formats and selecting bands from files (the ``band_specs`` argument).  Here is an example that uses ``select_from_file`` to load multi-band ``HDF4`` arrays:

.. code-block:: python

    from elm.readers import BandSpec
    from elm.readers.metadata_selection import meta_is_day
    band_specs = list(map(lambda x: BandSpec(**x),
            [{'search_key': 'long_name', 'search_value': "Band 1 ", 'name': 'band_1'},
             {'search_key': 'long_name', 'search_value': "Band 2 ", 'name': 'band_2'},
             {'search_key': 'long_name', 'search_value': "Band 3 ", 'name': 'band_3'},
             {'search_key': 'long_name', 'search_value': "Band 4 ", 'name': 'band_4'},
             {'search_key': 'long_name', 'search_value': "Band 5 ", 'name': 'band_5'},
             {'search_key': 'long_name', 'search_value': "Band 6 ", 'name': 'band_6'},
             {'search_key': 'long_name', 'search_value': "Band 7 ", 'name': 'band_7'},
             {'search_key': 'long_name', 'search_value': "Band 9 ", 'name': 'band_9'},
             {'search_key': 'long_name', 'search_value': "Band 10 ", 'name': 'band_10'},
             {'search_key': 'long_name', 'search_value': "Band 11 ", 'name': 'band_11'}]))
    HDF4_FILES = [f for f in glob.glob(os.path.join(ELM_EXAMPLE_DATA_PATH, 'hdf4', '*hdf'))
                  if meta_is_day(load_hdf4_meta(f))]
    data_source = {
        'sampler': select_from_file,
        'band_specs': band_specs,
        'args_list': HDF4_FILES,
    }

Alternatively, to train on a single HDF4 file, we could have done:

.. code-block:: python

    from elm.readers import load_array
    from elm.sample_util.metadata_selection import example_meta_is_day
    HDF4_FILES = [f for f in glob.glob(os.path.join(ELM_EXAMPLE_DATA_PATH, 'hdf4', '*hdf'))
                  if example_meta_is_day(load_hdf4_meta(f))]
    data_source = {'X': load_array(HDF4_FILES[0], band_specs=band_specs)}


Transformations
---------------

A ``Pipeline`` is created by giving a list of steps - the steps before the final step are known as transformers and the final step is the estimator.  See also the full docs on :doc:`elm.pipeline.steps<pipeline-steps>`.

 * Transformer steps must be taken from one of the classes in ``elm.pipeline.steps``. The purpose of ``elm.pipeline.steps`` is to wrap preprocessors and transformers from scikit-learn for use with :doc:`ElmStore<elm-store>`s or ``xarray.Dataset``s.

Here is an example ``Pipeline`` of transformations before K-Means

.. _xarray.DataArray: http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html

.. _StandardScaler: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

.. _sklearn.preprocessing: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
.. _sklearn.feature_selection: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection

.. _MiniBatchKMeans: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html

.. code-block:: python

    from elm.pipeline import steps, Pipeline
    pipeline_steps = [steps.Flatten(),
                      ('scaler', steps.StandardScaler()),
                      ('pca', steps.Transform(IncrementalPCA(n_components=4), partial_fit_batches=2)),
                      ('kmeans', MiniBatchKMeans(n_clusters=4, compute_labels=True)),]


The example above calls:

* ``steps.Flatten`` first (See :ref:`transformers-flatten`) first, as utility for flattening our multi-band raster HDF4 sample(s) into an :doc:`ElmStore<elm-store>` with a single `xarray.DataArray`_, called ``flat``, with each band as a column in ``flat``.
* `StandardScaler`_ with default arguments from ``sklearn.prepreprocessing`` (all other transformers from `sklearn.preprocessing`_ and `sklearn.feature_selection`_ are also attributes of ``elm.pipeline.steps`` and could be used here)
* PCA with ``elm.pipeline.steps.Transform`` to wrap scikit-learn transformers to allow multiple calls to ``partial_fit`` within a single fitting task of the final estimator - ``steps.Transform`` is initialized with:

  * A scikit-learn transformer as an argument
  * ``partial_fit_batches`` as a keyword, defaulting to 1. Note: using ``partial_fit_batches != 1`` requires a transformer with a ``partial_fit`` method
* Finally `MiniBatchKMeans`_


Multi-Model / Multi-Sample Fitting
----------------------------------

There are two multi-model approaches to fitting that can be used with a ``Pipeline``: :doc:`fit_ensemble<fit-ensemble>` or :doc:`fit_ea<fit-ea>`.  The examples above with a data source to a ``Pipeline`` and the transformation steps within one ``Pipeline`` instance work similarly in :doc:`fit_ensemble<fit-ensemble>` and :doc:`fit_ea<fit-ea>`.

Other similarities between :doc:`fit_ea<fit-ea>` and :doc:`fit_ensemble<fit-ensemble>` include the following common keyword arguments:
 * ``scoring`` a callable with a signature like ``elm.model_selection.kmeans.kmeans_aic`` (See :doc:`API docs<api>` ) or a string like ``f_classif`` attribute name from ``sklearn.metrics``
 * ``scoring_kwargs`` kwargs passed to the ``scoring`` callable if needed
 * ``saved_ensemble_size`` an integer indicating how many ``Pipeline`` estimators to retain in the final ensemble

Read more on controlling ensemble or evolutionary algorithm approaches to fitting:
 * :doc:`fit_ensemble<fit-ensemble>`
 * :doc:`fit_ea<fit-ea>`
 * :ref:`controlling-ensemble`

Multi-Model / Multi-Sample Prediction
-------------------------------------

After :doc:`fit_ensemble<fit-ensemble>` or :doc:`fit_ea<fit-ea>` has been called on a ``Pipeline`` instance, the instance will have the attribute ``ensemble`` a list of `(tag, pipeline)` tuples which are the final ``Pipeline`` instances selected by either of the fitting functions (see also ``saved_ensemble_size`` - See :ref:`controlling-ensemble`).  With a fitted ``Pipeline`` instance, :doc:`predict_many<predict-many>` can be called on the instance to predict from every ensemble member (``Pipeline`` instance) on a single ``X`` sample or from every ensemble member and every sample if ``sampler`` and ``args_list`` are given in place of ``X``.

Read more on controlling :doc:`predict_many<predict-many>`.
