# NASA SBIR Phase I - Open Source Parallel Image Analysis and Machine Learning Pipeline

Phase I - Techniques for
* Data mining
* Fusion
* Sub setting
* Discovery
* Visualization

Also in Phase I: Feasibility research on nonlinear dimensionality reduction and spectral unmixing methods for large data.

Phase II
 * Nonlinear dimensionality reduction and spectral unmixing
 * Feature extraction
 * Change detection

## Scope

Phase I

Flexible API where most functions can take an image, mosaic, or other array

Several unsupervised classification and clustering methods

Pre-processing options

Prediction

Cluster plugin(s) for:
    OpenGrADS, numpy, scipy, numba, NetCDF4 and HDF5
    dask+distributed, numba, xarray

Support for most operating systems/platforms on most parts of the API.

Phase I Milestones

Milestone I: Cluster plugin for dask+distributed stack

Milestone I Task 1: Streamlined deployment of dask distributed
This will install dask+distributed plus the stack we need for machine learning and
It will manage dask+distributed processes - which I think has been largely accomplished in: [https://docs.continuum.io/anaconda-cluster/plugins](https://docs.continuum.io/anaconda-cluster/plugins)

Milestone I Task 2: Automate installation of weather and satellite data tools on clusters
So our main focus is on writing cluster config's that install that dask+distributed plugin with the other weather data tools for image analysis: xarray, numba, numpy, scipy, sklearn, etc

Milestone II: Open source satellite data classification engine

Milestone II Task 1: API design to support the following input structures:

* A single image,
* A mosaic of images at similar points in time,
* Tiling or blocking of images or mosaics to run them through some API algorithms at multiple scales, and/or
* An image, mosaic, or tiled image that has been limited by an analysis mask,
* An image, mosaic, or tiled image with resampling or aggregation at a given spatial resolution,

Milestone II Task 2: Scalable Unsupervised Classification Model Fitting

Scikit-learn partial fit unsupervised methods in multi-class or one-vs-rest mode
* Kmeans
* Naive Bayes
* SGD
* other partial_fit methods of scikit-learn

Milestone II Task 3: Classification Prediction Engine

Support:
* Persistence
* Summarizing the predictions of a classifier in space and time
  * Areal extent of each class in each time step
* Differencing classification maps
* Classification diagnostics persisted with the predictor objects, such as ROC curves, confusion matrix, etc.

Milestone II Task 4: Pre-Processing Options

Support
* Limiting classification or prediction to a mask based on other arrays
* PCA
* Polynomial terms
* Scaling options like min/max scaler, z-score scaler
* Making an analysis mask based on a formula that refers to bands, e.g. NDVI thresholding as a layer to limit the domain of a classifier or predictor to urban or forest

Milestone II Task 5: Scheduler options for a variety of environments
Provide examples and testing of the image pipeline using a local dask scheduler or distributed one.

Milestone III:

Milestone III Task 1: Distribution and Promotion of Satellite Classification Tool
Milestone III Task 2: Documentation
Milestone III Task 3: Final Report on Phase I


## Peter Additional Notes May 2016

We have flexibility mentioned on input data types, but not on output data format.  The flexible API mentioned at the start of Milestone II should also consider a variety of output options, such as saving images of classification maps, including loading intermediate data or previous classifiers from a cache.

## Additional work identified for Phase I

Research how dask / xarray / numba can be used for nonlinear dimensionality reduction and spectral unmixing.  Begin by looking at [pysptools Pixel Purity Index and other methods](http://pysptools.sourceforge.net/_modules/pysptools/eea/eea.html#PPI) which work with in-memory numpy arrays.

# See also

 * [README on features of the new image pipeline](README_features.md)
 * [README on testing practices](README_testing.md)

