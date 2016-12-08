About Ensemble Learning Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensemble Learning Models (``elm`` ) is a set of tools for parallel ensemble unsupervised and supervised machine learning, with a focus on data structures useful in climate science and satellite imagery.

.. _dask-distributed: http://distributed.readthedocs.io/en/latest/

.. _xarray: http://xarray.pydata.org/en/stable/

.. _scikit-learn: http://scikit-learn.org/stable/

.. _estimator interface: http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator

.. _xarray data structures: http://xarray.pydata.org/en/stable/data-structures.html

``elm`` Capabilities
--------------------

* :doc:`Ensemble learning<fit-ensemble>`
* :doc:`Large scale prediction<predict-many>`
* :doc:`Genetic algorithms<fit-ea>`
* :doc:`Common preprocessing operations for satellite imagery and climate data<pipeline-steps>`

These capabilities are best shown in the

 * :doc:`Elm clustering introduction<clustering_example>`
 * :doc:`Other elm examples<examples>`
 * :doc:`Use cases<use-cases>`

``elm`` wraps together the following Python packages:

* `dask-distributed`_: ``elm`` uses `dask-distributed`_ for parallelism over ensemble fitting and prediction
* `scikit-learn`_ : ``elm`` can use unsupervised and supervised models, preprocessors, scoring functions, and postprocessors from ``scikit-learn`` or any estimator that follows the `scikit-learn` initialize / fit / predict `estimator interface`_.
* `xarray`_ : ``elm`` wraps `xarray data structures`_ for n-dimensional arrays, such as 3-dimensional weather cubes, and for collections of 2-D rasters, such as a LANDSAT sample

``elm`` is a Work in Progress
--------------------------
``elm`` is immature and largely for experimental use.

The developers do not promise backwards compatibility with future versions.

Next steps
----------

.. _Try the example notebooks: http://github.com/ContinuumIO/elm-examples

* :doc:`Install elm<install>`
* `Try the example notebooks`_

