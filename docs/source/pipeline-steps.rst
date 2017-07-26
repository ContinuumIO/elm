Customizable Pipeline Steps
===========================

The examples below assume you have created a random :doc:`ElmStore<elm-store>` as follows:

.. code-block:: python

    from elm.sample_util.make_blobs import random_elm_store
    X = random_elm_store()

Operations to reshape an :doc:`ElmStore<elm-store>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``Flatten`` - Flatten each 2-D ``DataArray`` in an :doc:`ElmStore<elm-store>` to create an :doc:`ElmStore<elm-store>` with a single ``DataArray`` called ``flat`` that is 2-D (each band raster is raveled from 2-D to a 1-D column in ``flat``).  Example:

.. code-block:: python

    steps.Flatten().fit_transform(X)

* ``Agg`` - Aggregate over a dimension or axis.  Example:

.. code-block:: python

    steps.Agg(axis=0, func='mean').fit_transform(X)

* ``DropNaRows`` - Remove null / NaN rows from an :doc:`ElmStore<elm-store>` that has been through ``steps.Flatten()``:

.. code-block:: python

    steps.DropNaRows().fit_transform(*steps.Flatten().fit_transform(X))

* ``InverseFlatten`` - Convert a flattened :doc:`ElmStore<elm-store>` back to 2-D rasters as separate ``DataArray`` values in an :doc:`ElmStore<elm-store>`.  Example:

.. code-block:: python

    steps.InverseFlatten().fit_transform(*steps.Flatten().fit_transform(X)


Use an unsupervised feature extractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _sklearn.decomposition: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition

* ``Transform`` - ``steps.Transform`` allows one to use any `sklearn.decomposition`_ method in an ``elm`` :doc:`Pipeline<pipeline>`.  Partial fit of the feature extractor can be accomplished by giving ``partial_fit_batches`` at initialization:

.. code-block:: python

    from sklearn.decomposition import IncrementalPCA
    X, y, sample_weight = steps.Flatten().fit_transform(X)
    pca = steps.Transform(IncrementalPCA(),
                    partial_fit_batches=2)
    pca.fit_transform(X)

Run a user-given callable
~~~~~~~~~~~~~~~~~~~~~~~~~
There are two choices for running a user-given callable in a :doc:`Pipeline<pipeline>` .  Using ``ModifySample`` is the most general, taking any shape of ``X``, ``y`` and ``sample_weight`` arguments, while ``FunctionTransformer`` requires that the :doc:`ElmStore<elm-store>` has been through ``steps.Flatten()``

* ``ModifySample`` - The following shows an example function with the required signature for use with ``ModifySample`` . It divides all the values in each ``DataArray`` by their sum.  Note the function always returns a tuple of ``(X, y, sample_weight)`` , even if ``y`` and ``sample_weight`` are not used by the function:

.. code-block:: python

    def modifier(X, y=None, sample_weight=None, **kwargs):
         for band in X.data_vars:
             arr = getattr(X, band)
             if kwargs.get('normalize'):
                 arr.values /= arr.values.max()
         return X, y, sample_weight

    steps.ModifySample(modifier, normalize=True).fit_transform(X)

.. _FunctionTransformer: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html

* ``FunctionTransformer`` - Here is an example using the `FunctionTransformer`_ from ``sklearn`` :

.. code-block:: python

    import numpy as np
    Xnew, y, sample_weight = steps.Flatten().fit_transform(X)
    Xnew, y, sample_weight = steps.FunctionTransformer(func=np.log).fit_transform(Xnew)

Preprocessing - Scaling / Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each of the following classes from scikit-learn have been wrapped for usage as a :doc:`Pipeline<pipeline>` step.  Each requires that the :doc:`ElmStore<elm-store>`

The examples below continue with ``Xnew`` a flattened :doc:`ElmStore<elm-store>` :

.. code-block:: python

    Xnew, y, sample_weight = steps.Flatten().fit_transform(X)

.. _KernelCenterer: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KernelCenterer.html

* ``KernelCenterer`` - See also `KernelCenterer`_ scikit-learn docs.

.. code-block:: python

    steps.KernelCenterer().fit_transform(Xnew)

.. _MaxAbsScaler: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html

* ``MaxAbsScaler`` -  See also `MaxAbsScaler`_ scikit-learn docs.

.. code-block:: python

    steps.MaxAbsScaler().fit_transform(*steps.Flatten().fit_transform(X))

.. _MinMaxScaler: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

* ``MinMaxScaler`` -  See also `MinMaxScaler`_ scikit-learn docs.

.. code-block:: python

    steps.MinMaxScaler().fit_transform(Xnew)

.. _Normalizer: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html

* ``Normalizer`` -  See also `Normalizer`_ scikit-learn docs.

.. code-block:: python

    steps.Normalizer().fit_transform(Xnew)

.. _RobustScaler: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html

* ``RobustScaler`` -  See also `RobustScaler`_ scikit-learn docs.

.. code-block:: python

    steps.RobustScaler().fit_transform(Xnew)

.. _PolynomialFeatures: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html

* ``PolynomialFeatures`` -  See also `PolynomialFeatures`_ scikit-learn docs.

.. code-block:: python

    step = steps.PolynomialFeatures(degree=3,
                                    interaction_only=False)
    step.fit_transform(Xnew)

.. _StandardScaler: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

* ``StandardScaler`` -  See also `StandardScaler`_ scikit-learn docs.

.. code-block:: python

    steps.StandardScaler().fit_transform(Xnew)

Encoding Preprocessors from ``sklearn``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each method here requires that the :doc:`ElmStore<elm-store>` has been through ``steps.Flatten()`` as follows:

.. code-block:: python

    Xnew, y, sample_weight = steps.Flatten().fit_transform(X)

.. _Binarizer: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html

* ``Binarizer`` - Binarize features.  See also `Binarizer`_ docs from ``sklearn`` .

.. code-block:: python

    steps.Binarizer().fit_transform(Xnew)

.. _Imputer: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html

* ``Imputer`` - Impute missing values.  See also `Imputer`_ docs from ``sklearn`` .

.. code-block:: python

    steps.Imputer().fit_transform(Xnew)

Feature selectors
~~~~~~~~~~~~~~~~~

.. _RFE: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
.. _RFECV: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
.. _SelectCanvas: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectCanvas.html
.. _SelectFdr: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html
.. _SelectFpr: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html
.. _SelectFromModel: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
.. _SelectFwe: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFwe.html
.. _SelectKBest: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
.. _SelectPercentile: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html
.. _VarianceThreshold: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html

The following list shows the feature selectors that may be used in a :doc:`Pipeline<pipeline>` .  The methods, with the exception of ``VarianceThreshold`` each require ``y`` to be not ``None``.

Setup for the examples:

.. code-block:: python

    X, y = random_elm_store(return_y=True)
    X = steps.Flatten().fit_transform(X)[0]

* ``RFE`` - See also `RFE`_  in ``sklearn`` docs. Example:

.. code-block:: python

    steps.RFE(estimator=LinearRegression()).fit_transform(X, y)

* ``RFECV`` - See also `RFECV`_   in ``sklearn`` docs. Example:

.. code-block:: python

    steps.RFECV(estimator=LinearRegression()).fit_transform(X, y)

* ``SelectFdr`` - See also `SelectFdr`_  in ``sklearn`` docs. Example:

.. code-block:: python

    steps.SelectFdr().fit_transform(X, y)

* ``SelectFpr`` - See also `SelectFpr`_  in ``sklearn`` docs. Example:

.. code-block:: python

    steps.SelectFpr().fit_transform(X, y)

* ``SelectFromModel`` - See also `SelectFromModel`_  in ``sklearn`` docs. Example:

.. code-block:: python

    steps.SelectFromModel(estimator=LinearRegression()).fit_transform(X, y)

* ``SelectFwe`` - See also `SelectFwe`_  in ``sklearn`` docs. Example:

.. code-block:: python

    steps.SelectFwe().fit_transform(X, y)

* ``SelectKBest`` - See also `SelectKBest`_  in ``sklearn`` docs. Example:

.. code-block:: python

    steps.SelectKBest(k=2).fit_transform(X, y)

* ``SelectPercentile`` - See also `SelectPercentile`_  in ``sklearn`` docs. Example:

.. code-block:: python

    steps.SelectPercentile(percentile=50).fit_transform(X, y)

* ``VarianceThreshold`` - See also `VarianceThreshold`_  in ``sklearn`` docs. Example:

.. code-block:: python

    steps.VarianceThreshold(threshold=6.92).fit_transform(X)

Normalizing time dimension of 3-D Cube
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following two functions take an :doc:`ElmStore<elm-store>` with a ``DataArray`` of any name that is a 3-D cube with a time dimension.  The functions run descriptive stats along the time dimension and flatten the spatial ``(x, y)`` dims to `space` (essentially a ``ravel`` of the ``(x, y)`` points).

Setup - make a compatible :doc:`ElmStore<elm-store>`:

.. code-block:: python

    from earthio import ElmStore
    import numpy as np
    import xarray as xr
    def make_3d():
        arr = np.random.uniform(0, 1, 100000).reshape(100, 10, 100)
        return ElmStore({'band_1': xr.DataArray(arr,
                                coords=[('time', np.arange(100)),
                                        ('x', np.arange(10)),
                                        ('y',np.arange(100))],
                                dims=('time', 'x', 'y'),
                                attrs={})}, attrs={}, add_canvas=False)
    X = make_3d()

* ``TSDescribe`` - Run ``scipy.stats.describe`` and other stats along the time axis of a 3-D cube ``DataArray`` .  Example:

.. code-block:: python

    s = steps.TSDescribe(band='band_1', axis=0)
    Xnew, y, sample_weight = s.fit_transform(X)
    Xnew.flat.band

The above code would show the ``band`` dimension of ``Xnew`` consists of different summary statistics, mostly from ``scipy.stats.describe`` :

.. code-block:: python

    <xarray.DataArray 'band' (band: 8)>
    array(['var', 'skew', 'kurt', 'min', 'max', 'median', 'std', 'np_skew'],
          dtype='<U7')
    Coordinates:
      * band     (band) <U7 'var' 'skew' 'kurt' 'min' 'max' 'median' 'std' 'np_skew'


* ``TSProbs`` - ``TSProbs`` will run bin, count and return probabilities associated with bin counts.  An example:

.. code-block:: python

    fixed_bins = steps.TSProbs(band='band_1',
                               bin_size=0.5,
                               num_bins=152,
                               log_probs=True,
                               axis=0)
    Xnew, y, sample_weight = fixed_bins.fit_transform(X)

The above would create the ``DataArray`` ``Xnew.flat`` with 152 columns consisting of the ``log`` transformed bin probabilities (152 bins of 0.5 width).

And the following would use irregular ( ``numpy.histogram`` ) bins rather than fixed bins and return probabilities without ``log`` transform first:

.. code-block:: python

    irregular_bins = steps.TSProbs(band='band_1',
                                   num_bins=152,
                                   log_probs=False,
                                   axis=0)
    Xnew, y, sample_weight = irregular_bins.fit_transform(X)

