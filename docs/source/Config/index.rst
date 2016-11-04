Job Configuration
========

This README walks through each section of the defaults.yaml default config and discusses how each section may be modified.

*Summary*: This file configures k-means to be run in ensemble with `partial_fit` for each model, taking samples
from the ladsweb ftp's VIIRS Level 2 data.

readers
--------

Each dict within the `readers` dict has the keys `load` and `bounds` which in turn point to
importable functions to load a file type and get its spatial bounds.  The key, `hdf4-eos` in this example,
is referenced in the `data_sources` section below.  In any case where a callable is required in the config, it is given in the format below as `package.subpackage.module:some_callable`.

.. code-block:: python 

    readers: {
      hdf4-eos: {
         load: "elm.readers.hdf4_L2_tools:load_hdf4",
         bounds: "elm.readers.hdf4_L2_tools:get_subdataset_bounds"
       }
      }

downloads
--------

This section names functions to be used for downloading data sources.  See also `data_sources` below.

.. code-block:: python 

    downloads: {
     default_downloader: "elm.acquire.ladsweb_ftp:main"
    }

data_sources
--------

`data_sources` name unique training or prediction related array data sets.  The key names, `NPP_DSRF1KD_L2GD` in this case,
are arbitrary but are used elsewhere (e.g. `pipeline`).  The entries in the dictionary are keywords to the `download` function,
which was defined above in `downloads`.  `years`, `data_days`, `product_number`, `product_name` are
arguments to the `download` function.  `band_specs` is a list of lists where each inner list consists of a
key match pattern for the metadata of a band, the value match pattern for that key / value in the band data, and the name
of the band when it is added as a column to the sample dataframe.  In this case, the dataframe will have
column names `band_1`, `band_2`, etc.

.. code-block:: python 

    data_sources: {
    NPP_DSRF1KD_L2GD: {
      product_number: 3001,
      product_name: NPP_DSRF1KD_L2GD,
      reader: hdf4-eos,
      file_pattern: "*.hdf",
      years: [2015],
      data_days: all,
      download: default_downloader,
      band_specs: [[long_name, "Band 1 ", band_1],
                  [long_name, "Band 2 ", band_2],
                  [long_name, "Band 3 ", band_3],
                  [long_name, "Band 4 ", band_4],
                  [long_name, "Band 5 ", band_5],
                  [long_name, "Band 7 ", band_7],
                  [long_name, "Band 8 ", band_8],
                  [long_name, "Band 10 ", band_10],
                  [long_name, "Band 11 ", band_11]]
            }
        }

file_generators
--------

`file_generators` is a dict where the keys are names of generators to reference elsewhere. The values are importable functions.
The default file generator is to iterate over all downloaded files for a given `product_name`, `product_number`, with those values taken
from the relevant entry in `data_sources`

.. code-block:: python 

    file_generators: {
      default_file_gen: "elm.readers.local_file_iterators:get_all_filenames_for_product",
    }

samplers
--------
`samplers` specifies a callable and its arguments for taking a sample from existing data for a data_source.  The keys
in `samplers`, e.g. `NPP_DSRF1KD_L2GD` below, are references to already-defined `data_sources`.  The default sampler `random_images_selection`
below can take arguments to form a sample from several files (`files_per_sample`), controlling the number of rows
in the sample (`n_rows_per_sample`).  `selection_kwargs` specify three functions `data_filter`, `metadata_filter` and `filename_filter`,
which, when given in a `package.module:func` format, provide filters on the data, metadata or filename, respectively.
`geo_filters` allows the sample to only be taken from area inside `include_polys` and outside `exclude_polys` if given.  The `include_polys`
and `exclude_polys` should be lists of strings, where the strings are keys in the `polys` dict (see below)

.. code-block:: python 

    samplers: {
      NPP_DSRF1KD_L2GD: {
        callable: "elm.samplers:random_images_selection",
        n_rows_per_sample: Null,
        files_per_sample: 1,
        file_generator: default_file_gen,
        selection_kwargs: {
          data_filter: Null,
          metadata_filter: Null,
          filename_filter: Null,
          geo_filters: {
            include_polys: [],
            exclude_polys: [],
          },
        }
      }
    }

polys
------
If using `include_polys` or `exclude_polys` as spatial filters, then provide specifications for how to load those polys (TODO
not developed yet).

.. code-block:: python 

    polys: {}

resamplers, aggregations, masks
------
These sections are not developed yet, but may include references to callables and their arguments which can be
used in the `pipeline` section for `on_each_sample`:

.. code-block:: python 

    resamplers: { 

    }

    aggregations: {

    }
    masks: {

    }

add_features
------
`add_features` allows new columns to be added to the output of one of the `samplers` of a data source.  The output of the sampler
function typically has the columns named in `band_specs` for the data source in which the sampler is used.  Giving a callable
in `add_features` can specify to run a function on the output of the sampler function, returning a data frame with new columns.
Examples would be to provide
 * Functions for NDVI and other common indices,
 * Ability to add all polynomial features of a given order, similar to what is done in `scikit-learn`'s polynomial features, and
 * A way to run any user-given function to add features


.. code-block:: python 

    add_features: {
      # an example of an entry here: NPP_DSRF1KD_L2GD_NDVI: "elm.preproc.add_features:ndvi",
    }

`train`
-------
`train` specifies training a model or ensemble with `partial_fit` if the model has a `partial_fit` method.

Specs:
 * `fit_func`: a function which takes a model and returns a fitted model, by default it is a `partial_fit`.
 * `model_selection_func`: a function which takes a list of models and returns a list of models in order of best to worst fit
 * `model_init_func`: a model initialization function, such as one from `scikit-learn` (kmeans by default)
 * `post_fit_func`: a function run on the output of `fit_func` (useful for adding attributes to the model needed by `model_selection_func`.
 `post_fit_func` takes arguments: model, df, kwargs, where `df` is the last sample of data as a dataframe.
 * `fit_kwargs`, `model_init_kwargs`, etc: Keyword arguments passed to each of the functions above
 * `sampler`: key from the `samplers` dict as a creator of each sample
 * `ml_features`: a list to indicate which columns of the final dataframe should be used for machine learning, defaulting to `all`.
 This may be useful if the dataframe contains columns useful to masking but not useful in training.
 * `output_tag`: to be implemented further - a system for tracking serialized trained models for later prediction
 * `data_source`: the data source to be used by the `sampler`

Example (the defaults)

.. code-block:: python 

    train: {
      kmeans: {
        fit_func: "elm.pipeline.partial_fit:partial_fit",
        model_selection_func: "elm.model_selection.kmeans:kmeans_model_averaging",
        model_init_func: "sklearn.cluster:MiniBatchKMeans",
        post_fit_func: "elm.model_selection.kmeans:kmeans_add_within_class_var",
        fit_kwargs: {
          n_batches: 2
        },
        model_init_kwargs: {
          compute_labels: True
        },
        model_selection_kwargs: {  # TODO this needs to be validated
          no_shuffle: 1,
        },
        ensemble_kwargs: {
          ensemble_size: 2,
          saved_ensemble_size: 1,
          n_generations: 2,
        },
        sampler: NPP_DSRF1KD_L2GD,
        ml_features: all,
        output_tag: kmeans,
        data_source: NPP_DSRF1KD_L2GD
      }
    }

`predict`
-------

This section will reference at `output_tag` from a prior `train` operation, either serialized or in the current pipeline, to make
predictions (applying classifiers to input data systematically).  It will include some options for common summaries like geographic
aggregation of predictions (e.g. areal extent of each class within each geographic region).


.. code-block:: python 

    predict: {
      kmeans: {
        from_output_tag: kmeans,
        file_generator: default_file_gen,
        poly_summarize: [],
      }
    }

change_detection
--------

To be developed, this section will refer to outputs from `predict` steps in the pipeline and difference them in time.

.. code-block:: python 

    change_detection: {
      time_series: []
    }

pipeline
------------

The config dictionaries shown above do not do anything until some parts of them are referenced in the `pipeline` list, where
each element of the list is a dictionary specifying an action.  An example here is to
 * Download the data source from the `data_sources` dict at the `NPP_DSRF1KD_L2GD` key according to the arguments in `data_sources`
 * Train the model using the `train` dictionary above at the key `kmeans`
 * Predict using the new `kmeans` model applying the trained classifier to every file from the `predict` dict's `file_generator`

.. code-block:: yaml 

    pipeline:
      - {download_data_sources: NPP_DSRF1KD_L2GD}
      - {train: kmeans}
      - {predict: kmeans}

Though not shown in the default `pipeline` above, a `train` or `predict` action in the `pipeline` list may also have a key called
`on_each_sample` to specify a series of filters and other callables to apply to a sample before training or prediction.  An example would be:

.. code-block:: yaml 

    pipeline:
      - {download_data_sources: NPP_DSRF1KD_L2GD}
      - {train: kmeans,
         on_each_sample:
          - {resampling: resampling_identifer_from_above}
          - {add_features: ndvi}
          - {aggregation: agg_identifier_from_above}

        }
      - {predict: kmeans}


Dask configuration
------------

The following are the defaults for dask settings in the config.  These can be overriden your config file or environment variables:


.. code-block:: bash 

    DASK_THREADS: Null #    int   os.cpu_count() if not given
    DASK_PROCESSES: Null  # int os.cpu_count() if not given
    DASK_EXECUTOR: SERIAL # SERIAL, DISTRIBUTED, PROCESS_POOL, THREAD_POOL
    DASK_SCHEDULER: Null  #  url if using DASK_EXECUTOR=DISTRIBUTED
    LADSWEB_LOCAL_CACHE: Null # where to download ftp data from ladsweb locally

Pipeline Config-Related Code
------------

See [the config subpackage](https://github.com/ContinuumIO/nasasbir/tree/master/elm/config) which has
 * [env.py](https://github.com/ContinuumIO/nasasbir/blob/master/elm/config/env.py) for parsing environment varibles
 * [load_config.py](https://github.com/ContinuumIO/nasasbir/blob/master/elm/config/load_config.py) which has `ConfigParser` a validator of config data structures
 * [cli.py](https://github.com/ContinuumIO/nasasbir/blob/master/elm/config/cli.py) which builds parsers for the interface
 * [defaults.py](https://github.com/ContinuumIO/nasasbir/blob/master/elm/config/defaults.py) which loads the default config shown in this README
 * [a test of the config loader](https://github.com/ContinuumIO/nasasbir/blob/d76a86969c070c10b3e358ea5ba533dc1206c959/elm/config/tests/test_config_simple.py)

.. toctree::
   :titlesonly:
