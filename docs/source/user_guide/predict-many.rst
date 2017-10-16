Multi-Model Prediction
======================

``elm``'s :doc:`predict_many<predict-many>` predicts for each estimator in a trained ensemble for one or more samples. :doc:`predict_many<predict-many>` takes some of the same data source keyword arguments that :doc:`fit_ea<fit-ea>` and :doc:`fit_ensemble<fit-ensemble>` use.  See also :doc:`Data Sources for a Pipeline<pipeline>` - it discusses using a single sample by giving the keyword arguments ``X`` or giving a ``sampler`` and ``args_list`` (list of unpackable args to the ``sampler`` callable).  The same logic applies for :doc:`predict_many<predict-many>`.

.. _xarray-pcolormesh: http://xarray.pydata.org/en/stable/generated/xarray.plot.pcolormesh.html

:doc:`predict_many<predict-many>` has a feature ``to_cube`` argument that is useful in prediction for spatial data.  ``to_cube=True`` (``True`` by default) means to convert the 1-D numpy array of predictions from the estimator of a :doc:`Pipeline<pipeline>` instance to a 2-D raster with the coordinates of the input data before the input data were flattened for training.  This makes it easy to make `xarray-pcolormesh`_ plots of predictions in spatial coordinates that are derived from models trained on spatial satellite and weather data.

.. _stochastic gradient descent: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier

Example - ``SGDClassifier``
---------------------------
The following example shows fitting a `stochastic gradient descent`_ classifier in ensemble with ``partial_fit``, varying the ``alpha`` and ``penalty`` parameters to ``sklearn.linear_model.SGDClassifier`` and finally predicting from the best models of the ensemble over several input samples.

Import from ``elm`` and ``sklearn``
-----------------------------------
This is a common set of ``import`` statements when working with ``elm``

.. code-block:: python

    from collections import OrderedDict
    from elm.pipeline import Pipeline, steps
    from earthio import *
    from sklearn.datasets import make_blobs
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score
    import numpy as np
    import xarray as xr

Define model selection
----------------------
We can define a callable with a signature like ``model_selection`` below to control which models are passed from generation to generation in an ensemble.  This function just uses ``best_idxes`` (Pareto sorted model fitness from the ``accuracy_score``):

.. code-block:: python

    def model_selection(models, best_idxes=None, **kwargs):
        top_n = kwargs['top_n']
        return [models[idx] for idx in best_idxes[:top_n]]

See also ``model_selection`` in :ref:`controlling-ensemble`.

Define initial ensemble
-----------------------
To vary the parameters of the initial ensemble of :doc:`Pipeline<pipeline>` instances, provide an ``ensemble_init_func`` .  ``pipe.new_with_params`` is used here to create a variety of :doc:`Pipeline<pipeline>` objects that have different ``SGDClassifier`` ``alpha`` and ``penalty`` parameters.

.. code-block:: python

    def ensemble_init_func(pipe, **kwargs):
        models = []
        for penalty in ('l1', 'l2'):
            for alpha in (0.0001, 0.001, 0.01):
                new_pipe = pipe.new_with_params(sgd__penalty=penalty, sgd__alpha=alpha)
                models.append(new_pipe)
        return models

See also ``ensemble_init_func`` in :ref:`controlling-ensemble`.

Control ``partial_fit`` and ensemble
-----------------------------------------------
The following ``dict`` are keywords to pass to :doc:`fit_ensemble<fit-ensemble>`, including setting the number of generations ``ngen``, using ``partial_fit`` twice per fitting of each model, and retaining finally the 2 best models (``saved_ensemble_size``).  Note also that ``partial_fit`` requires giving the keyword argument ``classes``, a sequence of all known classes, so this is passed via ``method_kwargs``:

.. code-block:: python

    ensemble_kwargs = {
        'model_selection': model_selection,
        'model_selection_kwargs': {
            'top_n': 2,
        },
        'ensemble_init_func': ensemble_init_func,
        'ngen': 3,
        'partial_fit_batches': 2,
        'saved_ensemble_size': 2,
        'method_kwargs': {'classes': np.arange(5)},
        'models_share_sample': True,
    }

See also ``ensemble_kwargs`` in :ref:`controlling-ensemble`.

Define a ``sampler``
-------------------------------------------------

The following lines of code use the synthetic data helper ``make_blobs`` from ``sklearn.datasets`` to create an ``ElmStore`` with 5 bands (each band is a ``DataArray`` )

.. code-block:: python


    rand_X_y = lambda n_samples: make_blobs(centers=[[1,2,3,4,5], [2,3,6,8,9], [3,4,5,10,12]], n_samples=n_samples)
    def sampler_train(width, height, **kwargs):
        X, y = rand_X_y(width * height)
        bands = ['band_{}'.format(idx + 1) for idx in range(X.shape[1])]
        es_data = OrderedDict()
        for idx, band in enumerate(bands):
            arr = xr.DataArray(X[:, idx].reshape(height, width),
                       coords=[('y', np.arange(height)),
                               ('x', np.arange(width))],
                       dims=('y', 'x'))
            es_data[band] = arr
        # No geo_transform in attrs of arr, so add_canvas = False
        es = ElmStore(es_data, add_canvas=False)
        sample_weight = None
        return es, y, sample_weight

Testing out ``sampler_train``:

.. code-block:: python

    In [42]: X, y, _ = sampler_train(10, 12)

    In [43]: X, y
    Out[43]:
    (ElmStore:
     <elm.ElmStore>
     Dimensions:  (x: 10, y: 12)
     Coordinates:
       * y        (y) int64 0 1 2 3 4 5 6 7 8 9 10 11
       * x        (x) int64 0 1 2 3 4 5 6 7 8 9
     Data variables:
         band_1   (y, x) float64 0.5343 -1.21 1.241 2.191 3.364 2.115 3.579 3.086 ...
         band_2   (y, x) float64 3.657 3.575 1.164 4.786 4.354 3.74 1.924 3.674 ...
         band_3   (y, x) float64 4.909 2.258 2.761 4.313 5.379 4.145 6.515 5.137 ...
         band_4   (y, x) float64 9.872 5.329 4.786 10.41 10.96 6.878 7.356 10.11 ...
         band_5   (y, x) float64 7.343 5.88 3.924 11.82 11.53 10.16 10.78 11.74 ...
     Attributes:
         _dummy_canvas: True
         band_order: ['band_1', 'band_2', 'band_3', 'band_4', 'band_5'],
     array([1, 0, 0, 2, 2, 1, 1, 2, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 2, 2, 0,
            0, 0, 2, 1, 0, 2, 0, 2, 2, 1, 2, 1, 2, 0, 2, 2, 0, 0, 2, 1, 1, 2, 2,
            0, 1, 2, 0, 1, 0, 1, 2, 0, 0, 0, 1, 1, 1, 2, 1, 1, 2, 2, 2, 0, 1, 1,
            2, 0, 2, 2, 1, 0, 1, 2, 1, 0, 0, 1, 1, 1, 2, 1, 0, 2, 1, 0, 1, 2, 0,
            0, 2, 1, 1, 0, 1, 2, 2, 1, 0, 2, 0, 1, 0, 1, 1, 2, 0, 0, 2, 1, 1, 1,
            2, 2, 1, 0, 2]))


:doc:`Pipeline<pipeline>` with scoring
-------------------------

The example below sets up ``accuracy_score`` for scoring a :doc:`Pipeline<pipeline>` that will flatten the sample and run ``SGDClassifier``.  The ``scoring_kwargs`` include ``greater_is_better`` (passed to ``sklearn.model_selection.make_scorer`` and ``score_weights`` determining whether sort models from minimum to maximum fitness (-1) or maximum to minimum (1).  Here we are maximimizing the ``accuracy_score``:

.. code-block:: python

    pipe = Pipeline([steps.Flatten(),
                     ('sgd', SGDClassifier())],
                     scoring=accuracy_score,
                     scoring_kwargs=dict(greater_is_better=True, score_weights=[1]))

Read more :doc:`documentation here<pipeline-steps>` on all the options available in ``elm.pipeline.steps`` .

Call :doc:`fit_ensemble<fit-ensemble>`
-------------------
Calling :doc:`fit_ensemble<fit-ensemble>` with an ``args_list`` of length 3, we are fitting all models in the ensemble to the same sample in one generation, then proceeding to fitting all models against the next sample in the next generation. In this case we have 3 generations (``ngen`` above) and 3 samples (``len(args_list)`` below) and ``models_share_sample=True``.  Each generation will have be a different sample and all models in a generation will be fitted to that sample.

.. code-block:: python

    data_source = dict(sampler=sampler_train, args_list=[(100, 120)] * 3)
    fitted = pipe.fit_ensemble(**data_source, **ensemble_kwargs)

Call :doc:`predict_many<predict-many>`
---------------------

We currently have 2 models in the ensemble (see ``saved_ensemble_size`` above that set the top N models to keep) and an ``args_list`` that will generate 3 samples: :doc:`predict_many<predict-many>` will predict 6 sample - model combinations.

.. code-block:: python

    preds = pipe.predict_many(**data_source)

Checking the number of predictions returned:

.. code-block:: python

    In [7]: len(preds)
    Out[7]: 6

Each item in ``preds`` is an ``ElmStore`` with a ``DataArray`` called ``predict``.  In this case that ``DataArray`` is a 2-D raster because we used the default keyword argument ``to_raster=True`` when :doc:`predict_many<predict-many>` was called.  The next snippet shows using the `plot` attribute of the ``predict`` ``DataArray``:

See also `documentation on plotting with xarray`_

.. _documentation on plotting with xarray: http://xarray.pydata.org/en/stable/plotting.html

.. code-block:: python

    p = preds[0]
    p.predict.plot.pcolormesh()

Predicting from an Ensemble Subset
----------------------------------
By default :doc:`predict_many<predict-many>` will look for an attribute on the :doc:`Pipeline<pipeline>` instance called ``.ensemble``, which is expected to be a list of ``(tag, pipeline)`` tuples, and predict from each trained :doc:`Pipeline<pipeline>` instance in ``.ensemble``.  Alternatively you can pass a list of ``(tag, pipeline)`` tuples as ``ensemble`` keyword argument.  The example below predicts only from the best model in the ensemble (the final ensemble is sorted by model score if ``scoring`` was given to :doc:`Pipeline<pipeline>` initialization). There are 3 predictions because there are 3 samples.

.. code-block:: python

    In [16]: subset = pipe.ensemble[:1]
    In [17]: preds = pipe.predict_many(ensemble=subset, **data_source)
    In [18]: len(preds)
    Out[18]: 3

Predictions Too Large For Memory
--------------------------------

In the examples above, :doc:`predict_many<predict-many>` has returned a list of ``ElmStore`` objects.  If the number of samples and/or models is large then keeping them all predictions in memory in a list is infeasible.  In these cases, pass a ``serialize`` argument (callable) to :doc:`predict_many<predict-many>` to serialize prediction ``ElmStore`` outputs as they are generated.  ``serialize`` should have a signature exactly like the example below:

.. code-block:: python

    import os
    from sklearn.externals import joblib
    def serialize(y, X, tag, elm_predict_path):
        dirr = os.path.join(elm_predict_path, tag)
        if not os.path.exists(dirr):
            os.mkdir(dirr) # assuming ELM_PREDICT_PATH in environment
        base = "_".join('{:.02f}'.format(_) for _ in sorted(X.canvas.bounds))
        joblib.dump(y, os.path.join(dirr, base + '.xr'))
        return X.canvas
    preds = pipe.predict_many(ensemble=pipe.ensemble[:1], serialize=serialize,**data_source)

In predicting over 3 samples and one model, we have created 3 ``joblib`` dump prediction files and returned three ``Canvas`` objects

.. code-block:: python

    In [27]: preds
    Out[27]:
    (Canvas(geo_transform=(-180, 0.1, 0, 90, 0, -0.1), buf_xsize=10, buf_ysize=10, dims=('y', 'x'), ravel_order='C', zbounds=None, tbounds=None, zsize=None, tsize=None, bounds=BoundingBox(left=-180.0, bottom=90.0, right=-179.1, top=89.1)),
     Canvas(geo_transform=(-180, 0.1, 0, 90, 0, -0.1), buf_xsize=10, buf_ysize=10, dims=('y', 'x'), ravel_order='C', zbounds=None, tbounds=None, zsize=None, tsize=None, bounds=BoundingBox(left=-180.0, bottom=90.0, right=-179.1, top=89.1)),
     Canvas(geo_transform=(-180, 0.1, 0, 90, 0, -0.1), buf_xsize=10, buf_ysize=10, dims=('y', 'x'), ravel_order='C', zbounds=None, tbounds=None, zsize=None, tsize=None, bounds=BoundingBox(left=-180.0, bottom=90.0, right=-179.1, top=89.1)))
    (Canvas(geo_transform=(-180, 0.1, 0, 90, 0, -0.1), buf_xsize=10, buf_ysize=10, dims=('y', 'x'), ravel_order='C', zbounds=None, tbounds=None, zsize=None, tsize=None, bounds=BoundingBox(left=-180.0, bottom=90.0, right=-179.1, top=89.1)),
     Canvas(geo_transform=(-180, 0.1, 0, 90, 0, -0.1), buf_xsize=10, buf_ysize=10, dims=('y', 'x'), ravel_order='C', zbounds=None, tbounds=None, zsize=None, tsize=None, bounds=BoundingBox(left=-180.0, bottom=90.0, right=-179.1, top=89.1)),
     Canvas(geo_transform=(-180, 0.1, 0, 90, 0, -0.1), buf_xsize=10, buf_ysize=10, dims=('y', 'x'), ravel_order='C', zbounds=None, tbounds=None, zsize=None, tsize=None, bounds=BoundingBox(left=-180.0, bottom=90.0, right=-179.1, top=89.1)))

Here are some notes on the arguments passed to ``serialize`` if given:

* `y` is an ``ElmStore`` either 1-D or 2-D (see ``to_raster`` keyword to :doc:`predict_many<predict-many>`)
* `X` is the ``X`` ``ElmStore`` that was used for prediction (the :doc:`Pipeline<pipeline>` will preserve ``attrs`` in ``X`` useful for serializing ``y`` as in the example above which used the `.canvas` attribute of ``X``)
* `tag` is a unique tag of sample and :doc:`Pipeline<pipeline>` instance
* `elm_predict_path` is the root dir for serialization output - ``ELM_PREDICT_PATH`` from :doc:`environment variables<environment-vars>`.

.. _dask-distributed: https://distributed.readthedocs.io/en/latest/quickstart.html#setup-dask-distributed-the-hard-way

Parallel Prediction
-------------------

To run :doc:`predict_many<predict-many>` (or :doc:`fit_ensemble<fit-ensemble>` or :doc:`fit_ea<fit-ea>`) in parallel using a dask-distributed client or dask ``ThreadPool`` client, use ``elm.config.client_context`` as shown here (continuing with the namespace defined by the snippets above)

First make sure you are running a ``dask-scheduler`` and ``dask-worker`` .  Read more here on `dask-distributed`_.

.. code-block:: python

    with client_context(dask_executor='DISTRIBUTED', dask_scheduler='10.0.0.10:8786') as client:
        ensemble_kwargs['client'] = client
        fitted = pipe.fit_ensemble(**data_source, **ensemble_kwargs)
        preds = pipe.predict_many(client=client, **data_source)

In the example above, ``client_context`` could have been called with no arguments if ``DASK_EXECUTOR`` and ``DASK_SCHEDULER`` :doc:`environment variables<environment-vars>`.

With parallel ``predict_many`` , each ensemble member / sample combination is a separate task - there is no parallelism within transformations of the ``Pipeline`` .
