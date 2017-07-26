Examples
========

This page provides examples of Python sessions using ``elm`` code and ``yaml`` config files that can be run with :doc:`elm-main<elm-main>`.

.. _Prerequisites:

Prerequisites for Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow the instructions for installation of ``elm`` and the ``elm-env`` conda environment ( :doc:`Install<install>` ):

* https://github.com/ContinuumIO/elm/tree/master/examples
* https://github.com/ContinuumIO/elm-data/

Also, define the environment variable ``ELM_EXAMPLE_DATA_PATH`` to be your full path to local clone of ``elm-data``

.. _notebooks-with-elm:

Jupyter (IPython) Notebooks with ``elm``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two notebooks provide worked out examples of using ``elm`` with time series of spatial weather data from NetCDF files.

The notebook then goes through how to build true and false color images with an :doc:`ElmStore<elm-store>` and ``matplotlib.pyplot.imshow``

.. _Clustering of temperature probability : https://github.com/ContinuumIO/elm/tree/master/examples/temperature-PDFs-clustering.ipynb

.. _Land cover clustering with K-Means : https://github.com/ContinuumIO/elm/tree/master/examples/LANDSAT_Example.ipynb

Notebooks using ``elm``
~~~~~~~~~~~~~~~~~~~~~~~

 * `Clustering of temperature probability`_ distributions in time
 * `Land cover clustering with K-Means`_, PCA, and other transformations

.. _Examples with SGD and K-Means in elm-examples: https://github.com/ContinuumIO/elm-examples/tree/master/configs

``yaml`` config files for ``elm-main``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 * `Examples with SGD and K-Means in Elm examples`_
