Why Elm?
~~~~~~~~

Ensemble Learning Models (elm) is a set of tools for creating multiple unsupervised and supervised machine learning models and training them in parallel on datasets too large to fit into the RAM of a single machine, with a focus on applications in climate science, GIS, and satellite imagery.

Some reasons for using elm over scikit-learn alone:

- Parallelize ML pipelines across the cores of a single machine or compute cluster
- Use out-of-core ML algorithms to process large datasets which are too large to fit into RAM
- Analyze multidimensional climate data, extending beyond the limitations of two-dimensional arrays and matrices
- Read data from file formats popular to climate science and GIS, such as netCDF, HDF4, HDF5, Shapefiles, GeoJSON, and GeoTIFF

More use-cases can be found `here <use-cases.html>`_.

``elm`` is a Work in Progress
--------------------------
``elm`` is immature and largely for experimental use.

The developers do not promise backwards compatibility with future versions.

Next steps
----------

.. _Try the example notebooks: https://github.com/ContinuumIO/elm/tree/master/examples

* :doc:`Use Cases for elm<use-cases>`
* :doc:`Install elm<install>`
* `Try the example notebooks`_

