# NASA SBIR Phase I - Open Source Parallel Image Analysis and Machine Learning Pipeline

## Using the Code

To use this code:

#### Install:
Create the development environment:
```
conda env create
```

Activate the environment:
```
source activate elm-env
```
(older versions of the code may have `elm` in place of `elm-env` above.  The environment name was changed to avoid conflict with `elm` package on anaconda.org.  The `elm-env` is uploaded to the nasasbir org on anaconda.org.)

Install the source:
```
python setup.py develop
```
Clone the `elm-data` repo using Git LFS so that more tests can be run:
```
brew install git-lfs # or apt-get, yum, etc
git lfs install
git clone http://github.com/ContinuumIO/elm-data
git remote add origin https//github.com/ContinuumIO/elm-data
```

Add the following to your .bashrc or environment, changing the paths depending on where you have cloned elm-data:
```
export DASK_EXECUTOR=SERIAL
export ELM_EXAMPLE_DATA_PATH=/Users/psteinberg/Documents/elm-data
```


#### Run the default config
```
DASK_EXECUTOR=SERIAL LADSWEB_LOCAL_CACHE=`pwd` DASK_SCHEDULER=1 elm-download-ladsweb --config elm/config/defaults/defaults.yaml
```
(replacing the yaml if not using the default VIIRS Level 2 dataset)

Run the faster running tests:
```
py.test -m "not slow"
```
or all of the tests:
```
py.test
```
or get the verbose test output
```
py.test -v
```
and cut and paste a test mark to run a specific test:
```
py.test -k test_train_makes_args_kwargs_ok
```

#### Run the default pipeline yaml:
_In serial_:
```
DASK_EXECUTOR=SERIAL elm-main --config elm/config/defaults/defaults.yaml  --echo-config
```
_With dask-distributed_:
```
dask-scheduler
```
In separate command prompts do this for each worker:
```
dworker 10.0.0.10:8786 # or what dask-scheduler gave as IP
```
Then
```
ELM_LOGGING_LEVEL=DEBUG DASK_EXECUTOR=DISTRIBUTED DASK_SCHEDULER=10.0.0.10:8786 elm-main --config elm/config/defaults/defaults.yaml  --echo-config
```
(You should modify the `ensembles` section of one of the configs in `elm/example_configs` to a larger ensemble to see a better parallel versus serial performance difference)

## Config File Format

It is easiest to copy the default config referenced above in snippets, and then follow [these instructions on editing the config](https://github.com/ContinuumIO/nasasbir/blob/master/README_config.md).

## About the NASA SBIR S5.03 Project
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

## Scope (Phase I)

* Flexible API where most functions can take an image, mosaic, or other array
* Several unsupervised classification and clustering methods for large data
* Pre-processing options
* Prediction
* Cluster plugin(s) or profiles for: OpenGrADS, numpy, scipy, numba, NetCDF4 and HDF5
    dask+distributed, numba, xarray

Support for most operating systems/platforms on most parts of the API.

### Phase I Milestones

#### Milestone I: Cluster plugin for dask+distributed stack

Milestone I Task 1: Streamlined deployment of dask distributed
This will install dask+distributed plus the stack we need for machine learning and
It will manage dask+distributed processes - which I think has been largely accomplished in: [https://docs.continuum.io/anaconda-cluster/plugins](https://docs.continuum.io/anaconda-cluster/plugins)

Milestone I Task 2: Automate installation of weather and satellite data tools on clusters
So our main focus is on writing cluster config's that install that dask+distributed plugin with the other weather data tools for image analysis: xarray, numba, numpy, scipy, sklearn, etc

#### Milestone II: Open source satellite data classification engine

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

#### Milestone III: Documentation, Reporting, and Promotion

Milestone III Task 1: Distribution and Promotion of Satellite Classification Tool
Milestone III Task 2: Documentation
Milestone III Task 3: Final Report on Phase I

## Additional work identified for Phase I

Research how dask / xarray / numba can be used for nonlinear dimensionality reduction and spectral unmixing.  Begin by looking at [pysptools Pixel Purity Index and other methods](http://pysptools.sourceforge.net/_modules/pysptools/eea/eea.html#PPI) which work with in-memory numpy arrays.

## Potential Changes to the Scope or Approach to Completing Scope

We have mentioned in the scope we will provide flexibility on input data types, such as images or mosaics, but we did not say anything about output data format, such as classification map images.  The flexible API mentioned at the start of Milestone II should also consider a variety of output options, such as saving images of classification maps, loading /saving cached predictor models, mapping bands of images to colors of output images, etc.

Another idea not mentioned in scope: as part of the flexible API for images/mosaics, we will have to address the problem of taking metadata like spatial / temporal bounds of an image from a filename as well as metadata about the bands.  There are cases where different bands are in different files or even folders and we may want to allow formation of machine learning input matrices based on bands from several sources (e.g. visible bands from one data set and infrared from another).

#### Notes from Matt Rocklin
 * We may have a much more computationally efficient and valuable parallel machine learning through ensemble approaches (many model fits) rather than parallel incremental learning.  An example would be separate solutions of a classification algorithm, then some model averaging logic above those separate solutions.  See [scikit-learn ensemble docs](http://scikit-learn.org/stable/modules/ensemble.html) for ideas.  This is potentially faster for very large data than training a single classifier incrementally.
 * For our testing data, we may consider just leaving it on an EBS volume that can be symlinked when needed, e.g. to a CI test box.
 * For putting n-d arrays on HDFS or S3 we should consider using [zarr](http://zarr.readthedocs.io/en/latest/) (zarr is under active development and subject to changes, but it is developed in loose collaboration with us)
 * In some cases, when creating an image mosaic object in our to-be-created flexible API, the spatial / temporal / band-related metadata will come from filenames and foldernames in some cases, but more often from metadata contained within the files (e.g. a GeoTiff file contains this information in the file, not the filename generally).
 * We should go over the tutorials on [dask delayed](http://dask.pydata.org/en/latest/delayed.html) which is dask for cases that are not clearly array or dataframe problems.

# Contribution Guidelines
 * Make a feature branch and PR in this repo (no forking)
 * Make [py.test tests in the package subdirectory(ies)](README_testing.md) you are modifying
 * Mark the PR with the tag WIP or ready for review, and mention the issue number that it is fixing
 * Use branch names like: `psteinberg/config-cli-cleanup` (person's name/what-it-does)
 * When something is ready for review, also mention it in flowdock
 * If there are reviewer comments to address on a PR, move the PR out of ready for review column and make into WIP status
 * Wait for at least one `LGTM` comment before merge of feature branch into master
 * Update the [waffle board](https://waffle.io/ContinuumIO/nasasbir) or use the tagging system in github

# See also

 * [README on features of the new image pipeline](README_features.md)
 * [README on testing practices](README_testing.md)
 * [README on NASA contacts and example datasets](README_nasa_projects.md)
 * [README on configuration of the image pipeline](https://github.com/ContinuumIO/nasasbir/blob/master/README_config.md)



