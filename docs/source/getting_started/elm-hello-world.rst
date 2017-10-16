Quick Start
===========

The following steps generate a visualization using `elm` on a synthetic dataset. As development on `elm` continues we strive to condense this document into a smaller example. For now, it offers insight into `elm`'s customizability and extensive feature set.

Step 1 - Choose Model(s)
~~~~~~~~~~~~~~~~~~~~~~~~

First import model(s) from scikit-learn and ``Pipeline`` and ``steps`` from ``elm.pipeline``:

.. testcode:: quick-start
    :pyversion: == 3.5

    from elm.config import client_context
    from earthio.filters.make_blobs import random_elm_store
    from elm.pipeline import Pipeline, steps
    from sklearn.decomposition import PCA
    from sklearn.cluster import AffinityPropagation

.. testoutput::
    :hide:

    <BLANKLINE>

.. _an xarray.Dataset: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html

* ``random_elm_store`` is a function that returns random rasters (``xarray.DataArray`` s) in an :doc:`ElmStore<elm-store>`, a data structure similar to `an xarray.Dataset`_
* ``steps`` is a module of :doc:`all the transformation steps possible in a Pipeline<pipeline-steps>`

See the :doc:`LANDSAT K-Means<clustering_example>` and :doc:`other examples<examples>` to see how to read an :doc:`ElmStore<elm-store>` from GeoTiff, ``HDF4``, ``HDF5``, or ``NetCDF``.

Step 2 - Define a ``sampler``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If fitting more than one sample, then define a ``sampler`` function to pass to :doc:`fit_ensemble<fit-ensemble>`.  Here we are using a partial of ``random_elm_store`` (synthetic data). If using a ``sampler`` , we also need to define ``args_list`` a list of tuples where each tuple can be unpacked as arguments to ``sampler``.  The length of ``args_list`` determines the number of samples potentially used.  Here we have 2 empty tuples as ``args_list`` because our ``sampler`` needs no arguments and we want 2 samples.  Alternatively the arguments ``X`` , ``y`` , and ``sample_weight`` may be given in place of ``sampler`` and ``args_list`` .

.. testcode:: quick-start
    :pyversion: == 3.5

    from functools import partial
    N_SAMPLES = 2
    layers = ['band_{}'.format(idx + 1) for idx in range(10)]
    sampler = partial(random_elm_store,
                      bands=layers,
                      width=60,
                      height=60)
    args_list = [(),] * N_SAMPLES


Step 2 - Define a :doc:`Pipeline<pipeline>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _sklearn.cluster.AffinityPropagation: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html

.. _sklearn.decomposition.PCA: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA

The code block below will use ``Flatten`` to convert each 2-D raster ( ``DataArray`` ) to give a single 1-D column in 2-D ``DataArray`` for machine learning.  The output of ``Flatten`` will be in turn passed to `sklearn.decomposition.PCA`_ and the reduced feature set from ``PCA`` will be passed to the `sklearn.cluster.AffinityPropagation`_ clustering model.

.. testcode:: quick-start
    :pyversion: == 3.5

    pipe = Pipeline([('flat', steps.Flatten()),
                     ('pca', steps.Transform(PCA())),
                     ('aff_prop', AffinityPropagation())])

Step 3 - Call :doc:`fit_ensemble<fit-ensemble>` with ``dask``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we can use :doc:`fit_ensemble<fit-ensemble>` to fit to one or more samples and one more instances of the ``pipe`` :doc:`Pipeline<pipeline>` above.  Below we are passing the ``sampler`` and ``args_list``, ``client``, which will be a ``dask-distributed`` or ``ThreadPool`` or None, depending on :doc:`environment variables<environment-vars>`. ``init_ensemble_size`` sets the number of :doc:`Pipeline<pipeline>` instances and ``models_share_sample=False`` means to fit all ``Pipeline`` / sample combinations (``2 X 2 == 4`` total members in this case).

.. testcode:: quick-start
    :pyversion: == 3.5

    pipe.fit_ensemble(sampler=sampler,
                      args_list=args_list,
                      init_ensemble_size=2,
                      models_share_sample=False,
                      ngen=1)

.. testoutput:: quick-start
    :pyversion: == 3.5
    :hide:
    :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

    Enter with ['band_1', 'band_2', 'band_3', 'band_4', 'band_5', 'band_6', 'band_7', 'band_8', 'band_9', 'band_10'] None None 60 60
    SHAPES 60 60 10 [[100 101 102 103 104 105 106 107 108 109]
     [110 111 112 113 114 115 116 117 118 119]
     [120 121 122 123 124 125 126 127 128 129]
     [130 131 132 133 134 135 136 137 138 139]
     [140 141 142 143 144 145 146 147 148 149]
     [150 151 152 153 154 155 156 157 158 159]
     [160 161 162 163 164 165 166 167 168 169]
     [170 171 172 173 174 175 176 177 178 179]
     [180 181 182 183 184 185 186 187 188 189]
     [190 191 192 193 194 195 196 197 198 199]] [[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]]
    Enter with ['band_1', 'band_2', 'band_3', 'band_4', 'band_5', 'band_6', 'band_7', 'band_8', 'band_9', 'band_10'] None None 60 60
    SHAPES 60 60 10 [[100 101 102 103 104 105 106 107 108 109]
     [110 111 112 113 114 115 116 117 118 119]
     [120 121 122 123 124 125 126 127 128 129]
     [130 131 132 133 134 135 136 137 138 139]
     [140 141 142 143 144 145 146 147 148 149]
     [150 151 152 153 154 155 156 157 158 159]
     [160 161 162 163 164 165 166 167 168 169]
     [170 171 172 173 174 175 176 177 178 179]
     [180 181 182 183 184 185 186 187 188 189]
     [190 191 192 193 194 195 196 197 198 199]] [[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]]

The code block with :doc:`fit_ensemble<fit-ensemble>` above would show the ``repr`` of the ``Pipeline`` object as follows:

.. doctest:: quick-start
    :pyversion: == 3.5

    >>> print(pipe)
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
        aff_prop: AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True,
                  damping=0.5, max_iter=200, preference=None, verbose=False)

We can confirm that we have ``4`` :doc:`Pipeline<pipeline>` instances in the trained ensemble:

.. doctest:: quick-start
    :pyversion: == 3.5

    >>> len(pipe.ensemble)
    4

Step 4 - Call :doc:`predict_many<predict-many>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`predict_many<predict-many>` will by default predict from the ensemble that was just trained (4 models in this case).  :doc:`predict_many<predict-many>` takes ``sampler`` and ``args_list`` like :doc:`fit_ensemble<fit-ensemble>`.  The ``args_list`` may differ from that given to ``fit_ensemble`` or be the same.  We have 4 trained models in the ``.ensemble`` attribute of ``pipe`` and 2 samples specified by ``args_list`` , so :doc:`predict_many<predict-many>` returns a list of 8 prediction :doc:`ElmStore<elm-store>`s

.. testcode:: quick-start
    :pyversion: == 3.5

    preds = pipe.predict_many(sampler=sampler, args_list=args_list)
    example = preds[0]

.. testoutput:: quick-start
    :pyversion: == 3.5
    :hide:
    :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

    Enter with ['band_1', 'band_2', 'band_3', 'band_4', 'band_5', 'band_6', 'band_7', 'band_8', 'band_9', 'band_10'] None None 60 60
    SHAPES 60 60 10 [[100 101 102 103 104 105 106 107 108 109]
     [110 111 112 113 114 115 116 117 118 119]
     [120 121 122 123 124 125 126 127 128 129]
     [130 131 132 133 134 135 136 137 138 139]
     [140 141 142 143 144 145 146 147 148 149]
     [150 151 152 153 154 155 156 157 158 159]
     [160 161 162 163 164 165 166 167 168 169]
     [170 171 172 173 174 175 176 177 178 179]
     [180 181 182 183 184 185 186 187 188 189]
     [190 191 192 193 194 195 196 197 198 199]] [[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]]
    Enter with ['band_1', 'band_2', 'band_3', 'band_4', 'band_5', 'band_6', 'band_7', 'band_8', 'band_9', 'band_10'] None None 60 60
    SHAPES 60 60 10 [[100 101 102 103 104 105 106 107 108 109]
     [110 111 112 113 114 115 116 117 118 119]
     [120 121 122 123 124 125 126 127 128 129]
     [130 131 132 133 134 135 136 137 138 139]
     [140 141 142 143 144 145 146 147 148 149]
     [150 151 152 153 154 155 156 157 158 159]
     [160 161 162 163 164 165 166 167 168 169]
     [170 171 172 173 174 175 176 177 178 179]
     [180 181 182 183 184 185 186 187 188 189]
     [190 191 192 193 194 195 196 197 198 199]] [[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]]

.. testcode:: quick-start
    :pyversion: == 3.5

    import matplotlib.pyplot as plt
    example.predict.plot.pcolormesh()
    plt.show()

.. testoutput:: quick-start
    :pyversion: == 3.5
    :hide:
    :options: +SKIP

-------------

**Read More** : :doc:`LANDSAT K-Means example<clustering_example>`
