from collections import namedtuple
import copy
from functools import partial
import logging
import inspect
import numpy as np

from elm.config import delayed, import_callable
from elm.pipeline.fit import fit
from concurrent.futures import as_completed

logger = logging.getLogger(__name__)

def wait_for_futures(futures, executor=None):
    if not executor:
        results = list(futures)
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
             model_init_class,
             model_init_kwargs,
             fit_func,
             fit_args,
             fit_kwargs,
             model_selection_func,
             model_selection_kwargs,
             **ensemble_kwargs):
    '''Train model(s) in ensemble

    Params:
        executor: None or a thread/process/distributed Executor
        model_init_class: a class which initializes a model
        model_init_kwargs: kwargs to the model init class
        fit_func: a function such as elm.pipeline.fit:fit
        fit_args: args to fit func, typically "action_data" as a tuple
                  (See also elm.pipeline.sample_pipeline which creates
                  action_data list, a list of sample pipeline actions)
        fit_kwargs: kwargs to the fit function such as kwargs related
                    to getting sample weights and corresponding Y values,
                    if needed
        model_selection_func: func which takes a model and kwargs
                    and returns an ordered list of models from
                    best to worst and may change the size of model
                    list
        ensemble_kwargs: kwargs such as "ensemble_size" and "n_generations"
                    which control the ensemble size and number of
                    generations in the ensemble (calls to model_selection_func)
        '''
    if hasattr(executor, 'map'):
        map_function = executor.map
        SERIAL_EVAL = True
    else:
        map_function = map
        SERIAL_EVAL = False
    if hasattr(executor, 'submit'):
        submit_func = executor.submit
    else:
        submit_func = no_executor_submit
    ensemble_size = ensemble_kwargs['ensemble_size']
    n_generations = ensemble_kwargs['n_generations']
    get_results = partial(wait_for_futures, executor=executor)
    models = [model_init_class(**model_init_kwargs) for _ in range(ensemble_size)]
    for generation in range(n_generations):
        logger.info('Ensemble generation {} of {}'.format(generation + 1, n_generations))
        args_kwargs = tuple(((model,) + tuple(fit_args), fit_kwargs)
                            for model in models)
        logger.debug('fit args_kwargs {}'.format(args_kwargs))
        models = get_results(
                    map_function(
                        lambda x: fit(*x[0], **x[1]),
                        args_kwargs
                    ))
        if generation < n_generations - 1:
            kwargs = copy.deepcopy(model_selection_kwargs)
            kwargs['generation'] = generation
            model_selection_func = import_callable(model_selection_func, True, model_selection_func)
            models = get_results(submit_func(model_selection_func, models, **kwargs))
    return models

