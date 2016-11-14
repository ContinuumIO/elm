``elm`` Intro
=============

This tutorial is a Hello World example with ``elm``

Step 1 - Choose Model(s)
~~~~~~~~~~~~~~~~~~~~~~~~

First import model(s) from scikit-learn and ``Pipeline`` and ``steps`` from ``elm.pipeline``:

.. code-block:: python

    from elm.config import client_context
    from elm.pipeline.tests.util import random_elm_store
    from elm.pipeline import Pipeline, steps
    from sklearn.decomposition import PCA
    from sklearn.cluster import AffinityPropagation

.. _an xarray.Dataset: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html

* ``random_elm_store`` is a function that returns random rasters (``xarray.DataArray`` s) in an :doc:`ElmStore<elm-store>`, a data structure similar to `an xarray.Dataset`_

See the :doc:`LANDSAT K-Means<clustering_example>` and :doc:`other examples<examples>` to see how to read an :doc:`ElmStore<elm-store>` from GeoTiff, ``HDF4``, ``HDF5``, or ``NetCDF``.

Step 2 - Define a ``sampler``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If fitting more than one sample, then define a ``sampler`` function to pass to :doc:`fit_ensemble<fit-ensemble>`.  Here we are using a partial of ``random_elm_store`` (synthetic). If using a ``sampler`` , we also need to define ``args_list`` a list of tuples where each tuple can be unpacked as arguments to ``sampler``.  The length of ``args_list`` determines the number of samples potentially used.  Here we have 2 empty tuples as ``args_list`` because our ``sampler`` needs no arguments and we want 2 samples.

.. code-block:: python

    from functools import partial
    N_SAMPLES = 2
    bands = ['band_{}'.format(idx + 1) for idx in range(10)]
    sampler = partial(random_elm_store,
                      bands=bands)
    args_list = [(),] * N_SAMPLES

Step 2 - Define a :doc:`Pipeline<pipeline>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _sklearn.cluster.AffinityPropagation: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html

.. _sklearn.decomposition.PCA: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA

The code block below will use :doc:`transform-flatten` to flatten (ravel) each raster ( ``DataArray`` ) to give a single 2-D ``DataArray`` for machine learning.  The output of ``Flatten`` will be in turn passed to `sklearn.decomposition.PCA`_ and the reduced feature set from ``PCA`` will be passed to the `sklearn.cluster.AffinityPropagation`_ clustering model.

.. code-block:: python

    pipe = Pipeline([('flat', steps.Flatten()),
                     ('pca', steps.Transform(PCA())),
                     ('aff_prop', AffinityPropagation())])

Step 3 - Call :doc:`fit_ensemble<fit-ensemble>` with ``dask``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we can use :doc:`fit_ensemble<fit-ensemble>` to fit to one or more samples and one more instances of the ``pipe`` :doc:`Pipeline<pipeline>` above.

.. code-block:: python

    with client_context() as client:
        pipe.fit_ensemble(sampler=sampler,
                          args_list=args_list,
                          client=client,
                          init_ensemble_size=2,
                          models_share_sample=False,
                          ngen=1)

In the logging, a message similar to this for each generation, showing the total number of calls to ``fit`` each generation:

.. code-block:: text

    Ensemble Generation 1 of 1: (2 members x 2 samples x 1 calls) = 4 fit calls this gen

The code block with :doc:`fit_ensemble<fit-ensemble>` above would show the ``repr`` of the ``Pipeline`` object as follows:

.. code-block:: text

    <elm.pipeline.Pipeline> with steps:
    flat: <elm.steps.Flatten>:

    pca: <elm.steps.Transform>:
        copy: True
        iterated_power: 'auto'
        n_components: None
        partial_fit_batches: None
        random_state: None
        svd_solver: 'auto'
        tol: 0.0
        whiten: False
    spectral: SpectralClustering(affinity='rbf', assign_labels='kmeans', coef0=1, degree=3,
              eigen_solver=None, eigen_tol=0.0, gamma=1.0, kernel_params=None,
              n_clusters=8, n_init=10, n_jobs=1, n_neighbors=10,
              random_state=None)

Step 4 - Call :doc:`predict_many<predict-many>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`predict_many<predict-many>` will by default predict from the ensemble that was just trained (2 models in this case).  :doc:`predict_many<predict-many>` takes ``sampler`` and ``args_list`` like :doc:`fit_ensemble<fit-ensemble>`.  The ``args_list`` may differ from that given to ``fit_ensemble``:

.. code-block:: python

    with client_context() as client:
        pipe.predict_many(sampler=sampler, args_list=args_list)


