Use Cases
=========

``elm`` (**Ensemble Learning Models**) is a versatile set of tools for ensemble and evolutionary algorithm approaches to training and selecting machine learning models and large scale prediction from trained models.  ``elm`` has a focus on data structures that are common in satellite and weather data analysis, such as rasters representing layers of satellite data or cubes of weather model output.

Common computational challenges in satellite and weather data machine learning include:

 * :ref:`large-scale-model`
 * :ref:`model-uncertainty`
 * :ref:`hyperparameterization`
 * :ref:`data-metadata-formats`
 * :ref:`preprocessing-input`
 * :ref:`predict-many-samples-models`

To address these challenges ``elm`` draws from existing Python packages:

.. _xarray: http://xarray.pydata.org/en/stable/

.. _scikit-learn: http://scikit-learn.org/stable/

.. _dask: http://dask.pydata.org/

.. _numba: http://numba.pydata.org/

.. _deap: https://deap.readthedocs.io/en/master/

.. _dask-distributed: http://distributed.readthedocs.io/en/latest/

.. _estimator interface: http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator

.. _xarray data structures: http://xarray.pydata.org/en/stable/data-structures.html

* `dask-distributed`_: ``elm`` uses `dask-distributed`_ for parallelism over ensemble fitting and prediction
* `scikit-learn`_ : ``elm`` can use unsupervised and supervised models, preprocessors, scoring functions, and postprocessors from ``scikit-learn`` or any estimator that follows the `scikit-learn` initialize / fit / predict `estimator interface`_.
* `xarray`_ : ``elm`` wraps `xarray data structures`_ for n-dimensional arrays, such as 3-dimensional weather cubes, and for collections of 2-D rasters, such as a LANDSAT sample

.. _large-scale-model:

Large-Scale Model Training
~~~~~~~~~~~~~~~~~~~~~~~~~~


``elm`` offers the following strategies for large scale training:

* Use of ``partial_fit`` for incremental training on series of saemples
* Ensemble modeling, training batches of models in generations in parallel, with model selection after each generation
* Use of a :doc:`Pipeline<pipeline>` with a sequence of :doc:`transformation steps<pipeline-steps>`
* ``partial_fit`` for incremental training of transformers used in ``Pipeline`` steps, such as PCA
* Custom user-given model selection logic in ensemble approaches to training

``elm`` can use ``dask`` to parallelize the activities above.

.. _model-uncertainty:

Model Uncertainty
~~~~~~~~~~~~~~~~~

Ensemble modeling can be used to account for uncertainty that arises from uncertain model parameters or uncertainty in the fitting process.  The ensemble approach in ``elm`` allows training and prediction from an ensemble where model parameters are varied, including parameters related to preprocessing transformations, such as feature selection or PCA transforms.  See the :doc:`predict_many<predict-many>` example.

.. _hyperparameterization:

Hyperparameterization / Model Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``elm`` offers two different algorithms for multi-model training with model selection:

 * :doc:`fit_ensemble<fit-ensemble>`: Running one batch of models at a time (a generation), running a user-given model selection function after each generation
 * :doc:`fit_ea<fit-ea>`: Using the NSGA-2 evolutionary algorithm to select best parameters for the best fit.

In either of these algorithms ``elm`` can use most of the model scoring features of ``scikit-learn`` or a user-given model scoring callable.

.. _scoring classes that work: http://scikit-learn.org/stable/modules/model_evaluation.html

See also:

 * :doc:`fit_ensemble<fit-ensemble>`
 * :doc:`fit_ea<fit-ea>`
 * ``elm.model_selection`` in :doc:`API docs<api>`
 * ``scikit-learn`` `scoring classes that work`_ with ``elm``

.. _data-metadata-formats:

Data/Metadata Formats
~~~~~~~~~~~~~~~~~~~~~
One challenge in satellite and weather data processing is the variety of input data formats, including GeoTiff, NetCDF, HDF4, HDF5, and others.  ``elm`` offers a function ``load_array`` which can load spatial array data in the following formats:

 * GeoTiff: Loads files from a directory of GeoTiffs, assuming each is a single-band raster
 * NetCDF: Loads variables from a NetCDF file
 * HDF4 / HDF5: Loads subdatasets from HDF4 and HDF5 files

``load_array`` creates an ``ElmStore`` (read more here), a fundamental data structure in ``elm`` that is essentially an ``xarray.Dataset`` with metadata standardization over the various file types.

.. _preprocessing-input:

Preprocessing Input Data
~~~~~~~~~~~~~~~~~~~~~~~~

``elm`` has a wide range of support for preprocessing activities.  One important feature of ``elm`` is its ability to train and/or predict from more than one sample and for each sample run a series of preprocessing steps that may include:

 * Scaling, adding polynomial features, or other preprocessors from ``sklearn.preprocessing``
 * Feature selection using any class from ``sklearn.feature_selection``
 * Flattening collections of rasters to a single 2-D matrix for fitting / prediction
 * Running user-given sample transformers
 * Resampling one raster onto another raster's coordinates
 * In-polygon selection
 * Feature extraction through transform models like PCA or ICA

See :doc:`elm.pipeline.steps<pipeline-steps>` for more information on preprocessing.

.. _predict-many-samples-models:

Predicting for Many Large Samples and/or Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``elm`` can use dask-distributed, a dask thread pool, or serial processing for predicting over a group (ensemble) of models and a single sample or series of samples.  ``elm``'s interface for large scale prediction, described here, is via the :doc:`predict_many<predict-many>` method of a ``Pipeline`` instance.


``elm`` Capabilities
--------------------

* :doc:`Ensemble learning<fit-ensemble>`
* :doc:`Large scale prediction<predict-many>`
* :doc:`Genetic algorithms<fit-ea>`
* :doc:`Common preprocessing operations for satellite imagery and climate data<pipeline-steps>`

These capabilities are best shown in the

 * :doc:`Elm introduction<elm-hello-world>`
 * :doc:`Elm clustering introduction<clustering_example>`
 * :doc:`Other elm examples<examples>`
 * :doc:`Use cases<use-cases>`
