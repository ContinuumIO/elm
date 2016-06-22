from collections import namedtuple
import numpy as np

from iamlp.settings import delayed, SERIAL_EVAL


@delayed
def ensemble(init_models,
            fit_function,
            model_selector,
            **ensemble_kwargs):
    models = None
    n_generations = ensemble_kwargs['n_generations']
    models = init_models(models=models)
    for generation in range(n_generations):
        models = [fit_function(model) for model in models]
        if not SERIAL_EVAL:
            models = models.compute()  #TODO is this the right place for .compute?
                                       # Need to pass in an executor?
        if generation < n_generations - 1:
            models = model_averaging(models, generation=generation)
        if not SERIAL_EVAL:
            models = [m.compute(get=get_func) for m in models.compute(get=get_func)]
    return models

