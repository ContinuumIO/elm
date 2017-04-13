## Ensemble Learning Models - Phase II Overview

Ensemble Learning Models combines open source Python tools to support large scale learning on satellite imagery and other Earth science data. The current phase of NASA funding for `elm` aims to:
 * Provide a wide variety of ensemble learning options, such as ensemble averaging, hierarchical modeling, and cross validation
 * Provide a variety of spatial preprocessors
 * Create a web-based user interface for common machine learning tasks with satellite imagery, such as labeling features
 * Develop several spectral clustering algorithms for large data
 * Promote `elm` for NASA projects and the satellite / climate science industry

## Milestones

The following milestones describe Phase II work over an 18 to 24 month period.

#### Milestone 1: ELM Architecture Review and Planning
 * Coordinate with NASA Goddard Space Flight Center on applications of ML
 * Review the state of `elm`, updates to `xarray`, `dask`, `dask-learn`
 * `elm` package rename requirement (currently has a conflict with Extreme Learning Machine)

#### Milestone 2: Improved Tools for Ensemble Fitting and Prediction
 * Cross Validation and Model Quality Assurance
 * Hierarchical Modeling
 * Vote Count Ensemble Averaging

#### Milestone 3: Zonal Statistics, Filters, and Change Detection
 * Zonal Statistics and Spatial Filters
 * Change Detection

#### Milestone 4: Improved Support for Spectral Clustering / Embedding and Manifold Learning
 * Develop 4 spectral clustering / embedding methods
 * More intuitive interface for scikit-learn spectral clustering methods

#### Milestone 5: Preliminary Web-Based Map User Interface for ELM
 * Bokeh Map Drawing Tools
 * Feature Labeling Tools
 * Visualizing Inputs and Predictions

#### Milestone 6: Data Structure Flexibility
 * Support for xarray’s multi-file datasets
 * Feature engineering options for 3-D and 4-D data sources
 * Allow numpy arrays or pandas data frames in place of ElmStores

#### Milestone 7: ELM User Interface Additional Features
 * Automated Simple Configurations of Elm
 * Runtime User Interface

#### Milestone 8: Promotion and Validation of ELM
 * Validation Through Realistic Examples
 * Climatic Regions Publication

#### Milestone 9: Hardening ELM
 * Hardening ELM
