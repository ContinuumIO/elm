from collections import namedtuple
import numpy as np

from iamlp.settings import delayed, SERIAL_EVAL
from iamlp.partial_fit import partial_fit
from iamlp.model_averaging.kmeans import kmeans_model_averaging


@delayed(pure=True)
def partial_fit_once(models, new_models, no_shuffle, partial_fit_kwargs):
    if models is not None:
        models = models[:no_shuffle] + new_models[no_shuffle:]
    else:
        models = new_models
    return [partial_fit(model, **partial_fit_kwargs) for model in models]


@delayed
def ensemble(init_models,
            output_tag,
            model_averaging,
            n_generations=2,
            no_shuffle=1,
            partial_fit_kwargs=None):
    models = None
    partial_fit_kwargs = partial_fit_kwargs or {}
    for generation in range(n_generations):
        new_models = init_models(models)
        models = partial_fit_once(models,
                                  new_models,
                                  no_shuffle,
                                  partial_fit_kwargs)
        if not SERIAL_EVAL:
            models = models.compute()
        if generation < n_generations - 1:
            models = model_averaging(models)
    return models

