# Features

## Cluster Plugins or Config Files

Create a cluster profile to automate installation of:

 * OpenGrADS
 * numpy
 * scipy
 * numba
 * NetCDF4
 * HDF5
 * pysptools
 * other Python tools identified by NASA

The cluster profile should use the dask plugin.

The API we develop should support either a local dask scheduler or distributed dask scheduler where feasible.

## Dask-Xarray-Numba Image Tools

From the proposal, the image pipeline will have "a flexible API where most functions support" the following input data types:

 * Single image - panchromatic, multi- or hyperspectral
 * Image file glob with metadata taken from filename/foldername regexes
 * Custom image generator function
 * HDFS Image file glob?
 * One of the data types above, with a mask that is:
   * An image or mosaic type like those listed above, and/or
   * Outside data like a DEM or NDVI map, and/or
   * An integer classification map, where the included areas correspond to a given integer

Create an Xarray-based data structure to represent each of the types above and allow classification, pre- and post-processing from there.

### Pipeline Preprocess

The pipeline API should allow the following optional steps:

 * Resampling - Bilinear 2-d interpolation of each band by default
 * Aggregation - Scalar summary statistic on coarser spatial grid
 * Create Mask - Use thresholding or expression on an output array to create a new mask (e.g. NDVI mask)
 * Scaling - max/min scaling, z-scores
 * Linear Dimensionality Reduction - PCA
   * sklearn.decomposition.IncrementalPCA
   * sklearn.cluster.MiniBatchKMeans
 * Polynomial Terms - Adding polynomial terms to an input matrix
 * Computation of user-defined features from functional specs.  For example, it may be necessary to calculate a ratio of different bands as an additional column for machine learning input.  NDVI is an example.
 * Thresholding for mask creation, e.g. limit a classifier to considering areas where NDVI > X.

### Pipeline Classifier/Regressors

The image pipeline should support [these classifiers with `partial_fit` methods](http://scikit-learn.org/stable/modules/scaling_strategies.html)

 * sklearn.naive_bayes.MultinomialNB
 * sklearn.naive_bayes.BernoulliNB
 * sklearn.linear_model.Perceptron
 * sklearn.linear_model.SGDClassifier
 * sklearn.linear_model.PassiveAggressiveClassifier

and these regression methods:

 * sklearn.linear_model.SGDRegressor
 * sklearn.linear_model.PassiveAggressiveRegressor

and k-means:

 * sklearn.cluster.MiniBatchKMeans

### Output Types:

 * sklearn predictor models for each method listed above like PCA, classifiers, regressors, clustering. (pickle)
 * Image / image mosaic in nested directory structure
   * Provide means of mapping bands to colors, e.g.
     * With PCA, PC1 -> red, PC2 -> green, PC3 -> blue
 * Integer classification map (e.g. vegetation type map)
 * Summaries of classification prediction maps like
   * Areal extent of each class, perhaps in combination with some other classification or mask like geographic boundaries,
   * Differencing several classification maps in time
 * Reporting on model skill / error
 * Other ideas?

