Use Cases
=========

``elm`` (**Ensemble Learning Models**) is a versatile set of tools for ensemble and evolutionary algorithm approaches to training and selecting machine learning models and large scale prediction from trained models.  ``elm`` has a focus on data structures that are common in satellite and weather data analysis, such as rasters representing bands of satellite data or cubes of weather model output.

Common computational challenges in satellite and weather data machine learning include:

 * Large Scale Model Training
 * Model Uncertainty
 * Hyperparameterization / Model Selection
 * Data/Metadata Formats
 * Preprocessing Input Data
 * Predicting for Many Large Samples and/or Models

To address these challenges ``elm`` draws from existing Python packages:

 * ``xarray``
 * ``scikit-learn``
 * ``dask``
 * ``numba``
 * ``deap`` - TODO LINKS FOR EACH,



Large-Scale Model Training
~~~~~~~~~~~~~~~~~~~~~~~~~~


``elm`` offers the following strategies for large scale training:

* Use of ``partial_fit`` for incremental training on series of saemples
* Ensemble modeling, training batches of models in generations in parallel, with model selection after each generation
* Use of a ``Pipeline`` - TODO LINK for sequences of transformations on samples
* ``partial_fit`` for incremental training of transformers used in ``Pipeline`` steps, such as PCA
* Custom user-given model selection logic in ensemble approaches to training

``elm`` can use ``dask`` to parallelize the activities above.

More reading:

 * ``Pipeline``
 * ``fit_ensemble``
 * ``fit_ea``
 * ``predict_many``
 * ``client_context``
 * Environment variables.

Model Uncertainty
~~~~~~~~~~~~~~~~~

Ensemble modeling can be used to account for uncertainty that arises from uncertain model parameters or uncertainty in the fitting process.  The ensemble approach in ``elm`` allows training and prediction from an ensemble where model parameters are varied, including parameters related to preprocessing transformations, such as feature selection or PCA transforms.  See the following examples of ensembles with diverse members:

 * Loikith et al notebook - TODO LINK
 * Stochastic gradient descent classifier in ensemble - TODO LINK ``predict_many`` example

Hyperparameterization / Model Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``elm`` offers two different algorithms for multi-model training with model selection:
 * ``fit_ensemble``: Running one batch of models at a time (a generation), running a user-given model selection function after each generation
 * ``fit_ea`` - TODO LINK: Using NSGA-2 TODO LINK evolutionary algorithm to select best parameters for the best fit.

In either of these algorithms ``elm`` can use most of the model scoring features of ``scikit-learn`` or a user-given model scoring callable.

See also: - TODO LINKS on each
 * ``fit_ensemble``
 * ``fit_ea``
 * ``elm.model_selection``
 * ``scikit-learn`` scoring classes that work with ``elm``

Data/Metadata Formats
~~~~~~~~~~~~~~~~~~~~~
One challenge in satellite and weather data processing is the variety of input data formats, including GeoTiff, NetCDF, HDF4, HDF5, and others.  ``elm`` offers a function ``load_array`` which can load spatial array data in the following formats:

 * GeoTiff: Loads files from a directory of GeoTiffs, assuming each is a single-band raster
 * NetCDF: Loads variables from a NetCDF file
 * HDF4 / HDF5: Loads subdatasets from HDF4 and HDF5 files

``load_array`` creates an ``ElmStore`` (read more here), a fundamental data structure in ``elm`` that is essentially an ``xarray.Dataset`` with metadata standardization over the various file types.

Preprocessing Input Data
~~~~~~~~~~~~~~~~~~~~~~~~

``elm`` has a wide range of support for preprocessing activities.  One important feature of ``elm`` is its ability to train and/or predict from more than one sample and for each sample run a series of preprocessing steps that may include:

TODO LINKS ON THE LIST BELOW
 * Scaling, adding polynomial features, or other preprocessors from ``sklearn.preprocessing``
 * Feature selection using any class from ``sklearn.feature_selection``
 * Flattening collections of rasters to a single 2-D matrix for fitting / prediction - TODO LINK steps.Flatten
 * Running user-given sample transformers (see also TODO LINK)
 * Resampling one raster onto another raster's coordinates
 * In-polygon selection - TODO LINK
 * Feature extraction through transform models like PCA or ICA

Predicting for Many Large Samples and/or Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``elm`` can use dask-distributed, a dask thread pool, or serial processing for predicting over a group (ensemble) of models and a single sample or series of samples.  ``elm``'s interface for large scale prediction, described here - TODO LINK in detail, is via the ``predict_many`` method of a ``Pipeline`` instance.


``elm`` - Work in Progress
~~~~~~~~~~~~~~~~
``elm`` is immature and largely for experimental use.

The developers do not promise backwards compatibility with future versions.
