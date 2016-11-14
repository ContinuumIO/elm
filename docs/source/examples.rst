Examples
========
This page provides examples of Python sessions using ``elm`` code and ``yaml`` config files that can be run with ``elm-main`` (TODO LINK elm-main).

Prerequisites for Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow the instructions for installation of ``elm`` and the ``elm-env`` conda environment- TODO LINK, then clone:

* https://github.com/ContinuumIO/elm-examples/
* https://github.com/ContinuumIO/elm-data/

Also, define the environment variable ``ELM_EXAMPLE_DATA_PATH`` to be your full path to local clone of ``elm-data``

Jupyter (IPython) Notebooks with ``elm``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two notebooks provide worked out examples of using ``elm`` with time series of spatial weather data from NetCDF files.

The notebook then goes through how to build true and false color images with an ``ElmStore`` and ``matplotlib.pyplot.imshow``


Scripts using ``elm``
~~~~~~~~~~~~~~~~~~~~~

 * A K-Means example with PCA preprocessing in ensemble - https://github.com/ContinuumIO/elm-examples/blob/master/scripts/api_example.py: This example shows:
   * ``Pipeline``
   * ``fit_ensemble`` and an example of a ``model_selection`` callable
   * ``predict_many`` for a series of samples
   * Using ``band_specs`` (``elm.readers.BandSpec``)
   * ``partial_fit`` within K-Means and within ``IncrementalPCA``
   * Preprocessing with ``sklearn.preprocessing.StandardScaler``
   * ``client_context`` for using dask clients

 * A K-Means example with feature selection in NSGA-2 - https://github.com/ContinuumIO/elm-examples/blob/master/scripts/api_example.py: This example shows:
   * ``Pipeline``
   * ``fit_ea``
   * ``predict_many``
   * ``client_context`` for using dask clients
   * A custom function to get dependent data ``y`` array for a given ``X`` sample ( ``elm.pipeline.steps.ModifySample`` - TODO LINK)
 * ``predict_many`` Example with stochastic gradient descent classification - TODO LINK: This example shows
   * How to make an ``ElmStore``
   * ``fit_ensemble`` with a series of samples
   * Flattening rasters to a single 2-d array (``elm.pipeline.steps.Flatten`` - TODO LINK)

 * Notebooks
  * Clustering of temperature probability distributions in time
  * Land cover clustering with K-Means, PCA, and other transformations
