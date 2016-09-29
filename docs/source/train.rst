Train section of config
========

The train section of the config specifies several base models which may be used in ensemble training or genetic algorithms.

Here is an example of a `train` config for K-Means:

.. code-block:: python 

    train: {
      kmeans: {
        model_init_class: "sklearn.cluster:MiniBatchKMeans",
        model_init_kwargs: {
          compute_labels: True
        },
        ensemble: small_ensemble,
        output_tag: kmeans,
        model_scoring: kmeans_aic,
        model_selection: kmeans_model_averaging,
      }
    }

The model configured above will be referenced by `"kmeans"`, the key in the `train` dict.
 * The `model_init_class` names a module:callable which can be used to instantiate a model.  It is expected that the models follow the typical scikit-learn pattern of instantiation with only keyword arguments and subsequent calls to `fit`, `transform`, or `partial_fit` with X and Y data as necessary.
 * `model_init_kwargs` are any keyword arguments which should be passed to new models in the ensemble or genetic algorithm.
 * `ensemble` refers to an ensemble that has been configured in the `ensembles` section of config (see below)
 * `output_tag` is how the model should be saved within the directory named by the environment variable `ELM_TRAIN_PATH`.
 * `model_scoring` is a reference to functions named in the `model_scoring` section of config which provides a place for specifying how to score models and arrive at an array or scalar score.
 * `model_selection` is a reference to functions named in the `model_selection` section of the config.

The following sections would also need to be in the config for the example "train" config above to work:

.. code-block:: python 

    model_selection: {
      kmeans_model_averaging: {
        kwargs: {
          drop_n: 4,
          init_n: 4,
          evolve_n: 4,
        },
        func: "elm.model_selection.kmeans:kmeans_model_averaging",
    }
    ensembles: {
      small_ensemble: {
        init_ensemble_size: 16,  # how many models to initialize at start
        saved_ensemble_size: 4, # how many models to serialize as "best"
        ngen: 4,       # how many model train/select generations
        partial_fit_batches: 4,     # how many partial_fit calls per train/select generation
      }
    }

The sections above create a model selection object called `kmeans_model_averaging` to be referenced in the `train` section.  The keywords given to the function say to drop `drop_n` (4) models from the ensemble, `evolve_n` (4) specifies the number of solutions to draw from a K-Means run on top of the K-Means of the individual models, and `init_n` specifies how many new models to initiate randomly on each ensemble generation.  The keyword names are arbitrary and may be tailored to the `func` that is named in the `model_selection` dictionary.

The `ensembles` section controls how many models are run in each batch of the ensemble process.  At the start `init_ensemble_size` models are initiated and on each generation are trained against the same input X data (and Y and sample weights if needed).  On the completion of each batch of N model fits or transforms, the `model_selection_func` is called to modify existing models, initialize new models or anything that returns a list of models.  After `ngen` generations, `saved_ensemble_size` number of models are serialized.  Since the list of models in the ensemble is sorted by model scores before serialization, setting `saved_ensemble_size` to N is a way to save the top N models in goodness of fit.

The `train` config may be combined with a `param_grid` for a genetic algorithm (LINK TODO)
