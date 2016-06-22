from collections import namedtuple
import copy
import inspect
import numpy as np

from iamlp.config import delayed, SERIAL_EVAL
from concurrent.futures import as_completed

def wait_for_futures(futures, executor=None):
    if not executor:
        results = futures
    elif hasattr(executor, 'gather'): # distributed
        from distributed import progress
        progress(futures)
        results = executor.gather(futures)
    else:
        results = []
        for fut in as_completed(futures):
            if fut.exception():
                raise ValueError(fut.exception())
            results.append(fut.result())
    return results

def no_executor_submit(func, *args, **kwargs):
    return func(*args, **kwargs)

def ensemble(executor,
             n_models,
             n_generations,
             init_model_func,
             init_model_kwargs,
             partial_fit_func,
             partial_fit_args,
             partial_fit_kwargs,
             model_selector_func,
             model_selector_kwargs):
    if hasattr(executor, 'map'):
        map_function = executor.map
    else:
        map_function = map
    if hasattr(executor, 'submit'):
        submit_func = executor.submit
    else:
        submit_func = no_executor_submit
    get_results = partial(wait_for_futures, executor=executor)
    n_generations = ensemble_kwargs['n_generations']
    models = [init_model_func(**init_model_kwargs) for _ in range(n_models)]
    for generation in range(n_generations):
        args_kwargs = tuple(((model,) + tuple(partial_fit_args), partial_fit_kwargs)
                            for model in models)
        models = get_results(
                    map_function(
                        lambda x: partial_fit_func(*x[0], **x[1]),
                        args_kwargs
                    ))
        if not SERIAL_EVAL:
            models = models.compute()  #TODO is this the right place for .compute?
                                       # Need to pass in an executor?
        if generation < n_generations - 1:
            kwargs = copy.deepcopy(model_selector_kwargs)
            kwargs['generation'] = generation
            models = get_results(submit_func(model_selector_func, models, **kwargs))
    return models

