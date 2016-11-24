``elm`` ``yaml`` Specs
======================

Workflows involving ensemble and evolutionary methods and :doc:`predict_many<predict-many>` can also be specified in a ``yaml`` config file for running with the :doc:`elm-main<elm-main>` console entry point.  The ``yaml`` config can refer to functions from ``elm`` or user-given packages or modules.  Read more the `yaml configuration file format here`_

.. _yaml configuration file format here: http://yaml.org/spec/1.2/spec.html

The repository `elm-examples`_ has a number of example ``yaml`` configuration files for GeoTiff and ``HDF4`` files as input to K-Means or stochastic gradient descent classifiers.

.. _elm-examples: http://github.com/ContinuumIO/elm-examples
.. _elm-data: http://github.com/ContinuumIO/elm-data
.. _elm-repo: http://github.com/ContinuumIO/elm

This page walks through each part of a valid ``yaml`` config.

``ensembles``
-------------
The ``ensembles`` section creates named dicts of keyword arguments to :doc:`fit_ensemble<fit-ensemble>`.  The example below creates ``example_ensemble``, an identifier we can use elsewhere in the config.  If passing the keyword ``ensemble_init_func`` in an ensemble here, then it should be given in `"package.subpackage.module:callable"` notation like a setup.py console entry point, e.g. ``"my_kmeans_module:make_ensemble"``.

.. code-block:: yaml

    ensembles: {
      example_ensemble: {
        init_ensemble_size: 1,
        saved_ensemble_size: 1,
        ngen: 3,
        partial_fit_batches: 2,
      },
    }


``data_sources``
----------------

The dicts in ``data_sources`` create named ``sampler``s with their keyword arguments.

In the config, ``args_list`` can be a callable.  In this case, it is ``iter_files_recursively`` a function which takes ``top_dir`` and ``file_pattern`` as arguments.  The filenames returned by ``iter_files_recursively`` will be filtered by ``example_meta_is_day`` an example function for detecting whether a satellite data file is night or day based on its metadata.  If ``args_list`` is callable, it should take a variable number of keyword arguments (``**kwargs``).

This examples creates ``ds_example`` which selects from files to get bands 1 through 6, iterating recursively over ``.hdf`` files in ``ELM_EXAMPLE_DATA_PATH`` from the environment (``env:SOMETHING`` means take ``SOMETHING`` from environment variables).

``band_specs`` in the data source are passed to ``elm.readers.BandSpec`` (See also :doc:`ElmStore<elm-store>` and :doc:`LANDSAT Example<clustering_example>` ) and determine which bands (subdatasets in this HDF4 case) to include in a sample.

.. code-block:: yaml

    data_sources: {
     ds_example: {
      sampler: "elm.sample_util.band_selection:select_from_file",
      band_specs: [{search_key: long_name, search_value: "Band 1 ", name: band_1},
      {search_key: long_name, search_value: "Band 2 ", name: band_2},
      {search_key: long_name, search_value: "Band 3 ", name: band_3},
      {search_key: long_name, search_value: "Band 4 ", name: band_4},
      {search_key: long_name, search_value: "Band 5 ", name: band_5},
      {search_key: long_name, search_value: "Band 6 ", name: band_6},],
      args_list: "elm.readers.local_file_iterators:iter_files_recursively",
      top_dir: "env:ELM_EXAMPLE_DATA_PATH",
      metadata_filter: "elm.sample_util.metadata_selection:example_meta_is_day",
      file_pattern: "\\.hdf",
     },
    }

See also :ref:`elm-store-from-file`

``model_scoring``
-----------------
Each dict in ``model_scoring`` has a ``scoring`` callable and the other keys/values are passed as ``scoring_kwargs``.  These in turn become the ``scoring`` and ``scoring_kwargs`` to initialize a ``Pipeline`` instance.  This example creates a scorer called ``kmeans_aic``

.. code-block:: yaml

    model_scoring: {
      kmeans_aic: {
        scoring: "elm.model_selection.kmeans:kmeans_aic",
        score_weights: [-1],
      }
    }

``transform``
-------------
This section allows using transform model, such as ``IncrementalPCA`` from ``sklearn.decomposition``.  ``model_init_kwargs`` can include any keyword argument to the ``model_init_class``, as well as ``partial_fit_batches`` (``partial_fit`` operations on each ``Pipeline`` ``fit`` or ``partial_fit``).

.. code-block:: yaml

    transform: {
      pca: {
        model_init_class: "sklearn.decomposition:IncrementalPCA",
        model_init_kwargs: {"n_components": 2, partial_fit_batches: 2},
      }
    }

``sklearn_preprocessing``
-------------------------

This section configures scikit-learn preprocessing classes (``sklearn.preprocessing``), such as ``PolynomialFeatures``, for use elsewhere in the config.  Each key is an identifer and each dictionary contains a ``method`` (imported from ``sklearn.preprocessing``) and keyword arguments to that ``method``.

.. code-block:: yaml

    sklearn_preprocessing: {
      min_max: {
        method: MinMaxScaler,
        feature_range: [0, 1],
      },
      poly2_interact: {
        method: PolynomialFeatures,
        degree: 2,
        interaction_only: True,
        include_bias: True,
      },
    }

``train``
---------
The ``train`` dict configures the final estimator in a ``Pipeline``, in this case ``MiniBatchKMeans``.  This example shows how to run that estimator with the ``example_ensemble`` keyword arguments from above, model scoring section from above (``kmeans_aic``), passing ``drop_n`` and ``evolve_n`` to the ``model_selection`` callable.

.. code-block:: yaml

    train: {
      train_example: {
        model_init_class: "sklearn.cluster:MiniBatchKMeans",
        model_init_kwargs: {
          compute_labels: True
        },
        ensemble: example_ensemble,
        model_scoring: kmeans_aic,
        model_selection: "elm.model_selection.kmeans:kmeans_model_averaging",
        model_selection_kwargs: {
          drop_n: 4,
          evolve_n: 4,
        }
      }
    }

``feature_selection``
---------------------
Each key in this section is an identifier and the each dict is a feature selector configuration, naming a ``method`` to be imported from ``sklearn.preprocessing`` and keyword arguments to that ``method``.

.. code-block:: yaml

    feature_selection: {
        top_half: {
            method: SelectPercentile,
            percentile: 50,
            score_func: f_classif
        }

    }

``run``
-------
The ``run`` section names fitting and prediction jobs to be done by using identifiers created in the config's dictionaries reviewed above.

About the ``run`` section:
 * It is a list of actions
 * Each action in the list is a dict
 * Each action should have the key ``pipeline`` that is a list of dictionaries specifying steps (analogous to the interactive session :doc:`Pipeline<pipeline>` )
 * Each action should have a ``data_source`` key pointing to one of the ``data_sources`` named above
 * Each action can have ``predict`` and/or ``train`` key/value with the value being one of the named ``train`` dicts above

.. code-block:: yaml

    run:
      - {pipeline: [{select_canvas: band_1},
          {flatten: True},
          {drop_na_rows: True},
          {sklearn_preprocessing: poly2_interact},
          {sklearn_preprocessing: min_max},
          {transform: pca}],
         data_source: ds_example,
         predict: train_example,
         train: train_example}


The example above showed a ``run`` configuration with a ``pipeline`` of transforms inclusive of flattening rasters, dropping null rows, adding polynomial interaction terms, min-max scaling, and PCA.

Valid steps for ``run`` - ``pipeline``
-----------------------------
This section shows all of the valid steps that can be a config's ``run`` - ``pipeline`` lists (items that could have appeared in teh ``pipeline`` list in preceding example).

**flatten**

Flattens 2-D rasters as separate ``DataArray``s to a single ``DataArray`` called ``flat`` in an :doc:`ElmStore<elm-store>`.

.. code-block:: yaml

    {flatten: True}

See also :ref:`transform-flatten`.

*See also:* :docs:`elm.pipeline.steps<pipeline-steps>`

**drop_na_rows**

Drops null rows from an ``ElmStore`` or ``xarray.Dataset`` with a ``DataArray`` called ``flat`` (often this step follows ``{flatten: True} in a ``pipeline``).

.. code-block:: yaml

    {drop_na_rows: True}

See also :ref:`transform-dropnarows`.

**modify_sample**

Provides a callable and optionally keyword arguments to modify ``X`` and optionally ``y`` and ``sample_weight``.  See example of interactive use of ``elm.pipeline.steps.ModifySample`` here - TODO LINK and the function signature for a ``modify_sample`` callable here - TODO LINK.  This example shows how to run ``normalizer_func`` imported from a package and subpackage, passing ``keyword_1`` and ``keyword_2``.

.. code-block:: yaml

    {modify_sample: "mypackage.mysubpkg.mymodule:normalizer_func", keyword_1: 4, keyword_2: 99}

See also ``ModifySample`` usage in a  :doc:`K-Means LANDSAT example<cluster_example>` .

**transpose**

Transpose the dimensions of the ``ElmStore``, like this example for converting from ``("y", "x")`` dims to ``("x", "y")`` dims.

.. code-block:: yaml

    {transpose: ['x', 'y']}

**sklearn_preprocessing**

If a config has a dict called ``sklearn_preprocessing`` as in the example above, then named preprocessors in that dict can be used in the ``run`` - ``pipeline`` lists as follows:

.. code-block:: yaml

    {sklearn_preprocessing: poly2_interact}

where ``poly2_interact`` is a key in ``sklearn_preprocessing``

*See also:* ``elm.pipeline.steps.PolynomialFeatures`` in :doc:`elm.pipeline.steps<pipeline-steps>`

**feature_selection**

If a config has a dict called ``feature_selection`` as in the example above, then named feature selectors there can be used in the ``run`` - ``pipeline`` section like this:

.. code-block:: yaml

    {feature_selection: top_half}

where ``top_half`` is a named feature selector in ``feature_selection``.

**transform**

Note the config's ``transform`` section configures transform models like PCA but they are not used unless the config's ``run`` - ``pipeline`` lists have a ``transform`` action (dict) in them.  Here is an example:

.. code-block:: yaml

    {transform: pca}

where ``pca`` is a key in the config's ``transform`` dict.
