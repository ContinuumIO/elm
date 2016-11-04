Use Cases
=========

``elm`` (**Ensemble Learning Models**) is a versatile set of tools for ensemble and evolutionary algorithm approaches to training and selecting machine learning models and large scale prediction from trained models.  ``elm`` has a focus on data structures that are common in satellite and weather data analysis, such as rasters representing bands of satellite data or cubes of weather model output.

Common computational challenges in satellite and weather data machine learning include:

 * Large scale model training
 * Accounting for model parameter uncertainty
 * Hyperparameterization and model selection
 * Inconsistency in data/metadata storage and processing
 * Preprocessing input data
 * Predicting for a large sample and more than one model

``elm`` draws from existing Python packages, namely ``xarray``, ``scikit-learn``, ``dask``, ``numba``, and ``deap`` - TODO LINKS FOR EACH, to address these challenges

Large-Scale Model Training
~~~~~~~~~~~~~~~~~~~~~~~~~~


``elm`` offers the following strategies for large scale training:

* Use of ``partial_fit`` for incremental training on series of samples
* Ensemble modeling, training batches of models in generations in parallel, with model selection after each generation
* Use of a ``Pipeline`` - TODO LINK for sequences of transformations on samples
* ``partial_fit`` for incremental training of transformers used in ``Pipeline`` steps, such as PCA
* Custom user-given model selection logic in ensemble approaches to training


Directory of NetCDF files
~~~~~~~~~~~~~~~~~~~~~~~~~

Hyperparameterization


Algorithm developer
~~~~~~~~~~~~~~~~~~~


Scikit-Learn User
~~~~~~~~~~~~~~~~~~~~~~~~~~~


Academic Cluster Administrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


