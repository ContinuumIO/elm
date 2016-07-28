from collections import namedtuple, Sequence
import copy
import inspect
import logging

from concurrent.futures import as_completed
from functools import partial
import numpy as np

from elm.config import delayed, import_callable
from elm.pipeline.fit import fit
from elm.pipeline.serialize import save_models_with_meta
from elm.pipeline.executor_util import (wait_for_futures,
                                        no_executor_submit)
from elm.model_selection.base import (base_selection, no_selection)
from elm.model_selection.scoring import score_one_model
from elm.model_selection.sorting import pareto_front

logger = logging.getLogger(__name__)

def _validate_ensemble_members(models):
    err_msg = ('Failed to instantiate models as sequence of tuples '
               '(name, model) where model has a fit or '
               'partial_fit method.  ')
    if not models or not isinstance(models, Sequence):
        raise ValueError(err_msg + "Got {}".format(repr(models)))
    example = 'First item in models list: {}'.format(models[0])
    err_msg += example
    if not all(len(m) == 2 and isinstance(m, tuple) for m in models):
        raise ValueError(err_msg)
    return models

def ensemble(executor,
             model_init_class,
             model_init_kwargs,
             fit_args,
             fit_kwargs,
             model_scoring,
             model_scoring_kwargs,
             model_selection_func,
             model_selection_kwargs,
             **ensemble_kwargs):
    '''Train model(s) in ensemble

    Params:
        executor: None or a thread/process/distributed Executor
        model_init_class: a class which initializes a model
        model_init_kwargs: kwargs to the model init class
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
    else:
        map_function = map
    if hasattr(executor, 'submit'):
        submit_func = executor.submit
    else:
        submit_func = no_executor_submit
    model_selection_kwargs = model_selection_kwargs or {}
    ensemble_size = ensemble_kwargs['init_ensemble_size']
    n_generations = ensemble_kwargs['n_generations']
    get_results = partial(wait_for_futures, executor=executor)
    model_names = ensemble_kwargs.get('model_names', None)
    ensemble_init_func = ensemble_kwargs.get('ensemble_init_func') or None
    if not ensemble_init_func:
        models = tuple(('tag_{}'.format(idx), model_init_class(**model_init_kwargs))
                       for idx in range(ensemble_kwargs['init_ensemble_size']))
    else:
        ensemble_init_func = import_callable(ensemble_init_func)
        models = ensemble_init_func(model_init_class,
                                    model_init_kwargs,
                                    ensemble_kwargs)
    models = _validate_ensemble_members(models)
    if model_selection_func and model_selection_func != 'no_selection':
        model_selection_func = import_callable(model_selection_func)
    else:
        model_selection_func = no_selection
    fit_kwargs = copy.deepcopy(fit_kwargs)
    fit_kwargs['scoring'] = model_scoring
    fit_kwargs['scoring_kwargs'] = model_scoring_kwargs
    for generation in range(n_generations):
        model_names = [name for name, model in models]
        logger.info('Ensemble generation {} of {}'.format(generation + 1, n_generations))
        args_kwargs = tuple(((model,) + tuple(fit_args), fit_kwargs)
                            for name, model in models)
        logger.debug('fit args_kwargs {}'.format(args_kwargs))
        fitted = get_results(
                    map_function(
                        lambda x: fit(*x[0], **x[1]),
                        args_kwargs
                    ))
        models = tuple(zip(model_names, fitted))
        if model_selection_func:
            model_selection_kwargs['n_generations'] = n_generations
            model_selection_kwargs['generation'] = generation
            score_weights = model_selection_kwargs.get('score_weights') or None
            scoring_kwargs = fit_kwargs.get('scoring_kwargs') or {}
            key = 'greater_is_better'
            if key in scoring_kwargs:
                if not isinstance(score_weights, Sequence):
                    score_weights = [ 1 if scoring_kwargs[key] else -1]
            sort_fitness = model_selection_kwargs.get('sort_fitness') or None
            if not sort_fitness:
                sort_fitness = pareto_front
            else:
                sort_fitness = import_callable(sort_fitness)
            models = base_selection(models,
                                    model_selection_func=model_selection_func,
                                    sort_fitness=sort_fitness,
                                    score_weights=score_weights,
                                    **model_selection_kwargs)
            models = _validate_ensemble_members(models)
        else:
            pass # just training all ensemble members
                 # without replacing / re-ininializing / editing
                 # the model params

    if ensemble_kwargs.get('saved_ensemble_size') is not None:
        saved_models = models[:ensemble_kwargs['saved_ensemble_size']]
    else:
        saved_models = models
    model_paths, meta_path = save_models_with_meta(saved_models,
                                 ensemble_kwargs['config'].ELM_TRAIN_PATH,
                                 ensemble_kwargs['tag'],
                                 ensemble_kwargs['config'])
    logger.info('Created model pickles: {} '
                'and meta pickle {}'.format(model_paths, meta_path))
    return models


