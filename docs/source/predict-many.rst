Predict Many
============

``elm``'s ``predict_many`` predicts for each estimator in a trained ensemble for one or more samples. ``predict_many`` takes some of the same data source keyword arguments that ``fit_ea`` and ``fit_ensemble`` use.  See also Data Sources for a ``Pipeline``(TODO LINK to that section in pipeline.rst) - it discusses using a single sample by giving the keyword arguments ``X`` or giving a ``sampler`` and ``args_list`` (list of unpackable args to the ``sampler`` callable).  The same logic applies for ``predict_many``.

``predict_many`` has a feature ``to_cube`` argument that is useful in prediction for spatial data.  ``to_cube=True`` (True by default) means to convert the 1-D numpy array of predictions from the estimator of a ``Pipeline`` instance to a 2-D raster with the coordinates of the input data before the input data were flattened for training.  This makes it easy to make ``pcolormesh`` plots of predictions in spatial coordinates that are derived from models trained on spatial satellite and weather data.



Example - ``SGDClassifier``
---------------------------
The following example shows fitting a stochastic gradient descent classifier in ensemble, varying the ``alpha`` and ``penalty`` parameters to ``sklearn.linear_model.SGDClassifier``, then predicting from the best models over several input samples.

Import from ``elm`` and ``sklearn``
-----------------------------------

.. code-block:: python

    from collections import OrderedDict
    from elm.pipeline import Pipeline, steps
    from elm.readers import *
    from sklearn.datasets import make_blobs
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score
    import numpy as np
    import xarray as xr

Define model selection
----------------------

.. code-block:: python

    def model_selection(models, best_idxes=None, **kwargs):
        top_n = kwargs['top_n']
        return [models[idx] for idx in best_idxes[:top_n]]

Define initial ensemble
-----------------------

.. code-block:: python

    def ensemble_init_func(pipe, **kwargs):
        models = []
        for penalty in ('l1', 'l2'):
            for alpha in (0.0001, 0.001, 0.01):
                new_pipe = pipe.new_with_params(sgd__penalty=penalty, sgd__alpha=alpha)
                models.append(new_pipe)
        return models

Control ``partial_fit`` and ensemble
-----------------------------------------------

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
    }

Define a ``sampler``
-------------------------------------------------

.. code-block:: python


    rand_X_y = lambda n_samples: make_blobs(centers=[[1,2,3,4,5], [2,3,6,8,9], [3,4,5,10,12]], n_samples=n_samples)
    def sampler_train(width, height, **kwargs):
        X, y = rand_X_y(width * height)
        bands = ['band_{}'.format(idx + 1) for idx in range(X.shape[1])]
        es_data = OrderedDict()
        for idx, band in enumerate(bands):
            arr = xr.DataArray(X[:, idx].reshape(height, width), coords=[('y', np.arange(height)), ('x', np.arange(width))], dims=('y', 'x'))
            es_data[band] = arr
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


``Pipeline`` with scoring
-------------------------

.. code-block:: python

    pipe = Pipeline([steps.Flatten(), ('sgd', SGDClassifier())], scoring=accuracy_score, scoring_kwargs=dict(greater_is_better=True, score_weights=[1]))

Call ``fit_ensemble``
-------------------

.. code-block:: python

    data_source = dict(sampler=sampler_train, args_list=[(100, 120)] * 3)
    fitted = pipe.fit_ensemble(**data_source, **ensemble_kwargs)

Call ``predict_many``
---------------------

.. code-block:: python

    preds = pipe.predict_many(**data_source)

.. code-block:: python

    In [7]: len(preds)
    Out[7]: 6

Plotting the first one, using the `plot` attribute of the ``predict`` ``DataArray``:

.. code-block:: python

    p = preds[0]
    p.predict.plot.pcolormesh()

preds = pipe.predict_many(ensemble=pipe.ensemble[:1], **data_source)
In [18]: len(preds)
Out[18]: 3

Parallel Prediction
-------------------

To run ``predict_many`` (or ``fit_ensemble`` or ``fit_ea``) in parallel using a dask-distributed client or dask ``ThreadPool`` client, use ``elm.config.client_context`` as shown here (continuing with the namespace defined by the snippets above)

.. code-block:: python

    with client_context(dask_executor='DISTRIBUTED', dask_scheduler='10.0.0.10:8786') as client:
        ensemble_kwargs['client'] = client
        fitted = pipe.fit_ensemble(**data_source, **ensemble_kwargs)
        preds = pipe.predict_many(client=client, **data_source)

In the example above, ``client_context`` could have been called with no arguments if ``DASK_EXECUTOR`` and ``DASK_SCHEDULER`` environment variables were defined.  See also environment variables - TODO LINK
