from functools import partial, wraps
from itertools import chain, product
import copy
import logging
import numbers
import random

import dask

from elm.config import import_callable
from elm.config.dask_settings import wait_for_futures
from elm.pipeline.fit_and_score import fit_and_score
from elm.pipeline.serialize import (load_models_from_tag,
                                    serialize_models as _serialize_models)
from elm.pipeline.util import (make_model_args_from_config,
                               _prepare_fit_kwargs,
                               _validate_ensemble_members,
                               _get_model_selection,
                               _run_model_selection,
                               _ensemble_ea_prep,
                               _next_name)
from elm.sample_util.samplers import make_one_sample
from elm.pipeline.fit_and_score import fit_and_score

logger = logging.getLogger(__name__)

__all__ = ['ensemble']


def _fit_once(method, model, fit_score_kwargs, args):
    X, y, sample_weight = args
    return fit_and_score(model, X, y=y,
                         sample_weight=sample_weight,
                         **fit_score_kwargs)


def _run_fit_and_score_dask(dsk,
                            models,
                            fit_score_kwargs,
                            get_func,
                            models_share_sample,
                            method='fit',
                            sample_key=None):

    # Samples before Pipeline actions applied
    sample_keys = tuple(dsk)
    model_keys = [_[0] for _ in models]
    collect_keys = []
    if not models_share_sample:
        for (key, model), arg in zip(models, sample_keys):
            name = _next_name()
            dsk[name] = (partial(_fit_once, method, model, fit_score_kwargs), arg)
            collect_keys.append(name)
    else:
        sample_key = sample_key or random.choice(sample_keys)
        for key, model in models:
            name = _next_name()
            dsk[name] = (partial(_fit_once, method, model, fit_score_kwargs), sample_key)
            collect_keys.append(name)
    def tuple_of_args(*args):
        return tuple(args)
    new_models_name = _next_name()
    dsk[new_models_name] = (tuple_of_args, *collect_keys)
    if get_func is None:
        new_models = tuple(dask.get(dsk, new_models_name))
    else:
        new_models = tuple(get_func(dsk, new_models_name))
    models = tuple(zip(keys, new_models))
    return models


def _make_sample(pipe, args, sampler):
    return pipe.create_sample(X=None, y=None, sample_weight=None,
                              sampler=sampler, sampler_args=args)

def make_samples(pipe, args_gen, sampler):
    dsk = {}
    if callable(args_gen):
        args_gen = args_gen()
    for arg in args_gen:
        sample_name = _next_name()
        dsk[sample_name] = (_make_sample, pipe, arg, sampler)
    return dsk


def ensemble(pipe,
             ngen,
             X=None,
             y=None,
             sample_weight=None,
             sampler=None,
             args_gen=None,
             client=None,
             init_ensemble_size=1,
             ensemble_init_func=None,
             models_share_sample=True,
             model_selection=None,
             model_selection_kwargs=None,
             scoring=None,
             scoring_kwargs=None,
             method='fit',
             partial_fit_batches=1,
             classes=None,
             serialize_models=None,
             **kwargs):

    '''Train model(s) in ensemble

    '''
    from elm.config.dask_settings import get_func
    if hasattr(client, 'map'):
        map_function = client.map
    else:
        map_function = map

    get_results = partial(wait_for_futures, client=client)
    model_selection_kwargs = model_selection_kwargs or {}
    ensemble_size = init_ensemble_size or 1
    if not ensemble_init_func:
        models = tuple(copy.deepcopy(pipe) for _ in range(ensemble_size))
    else:
        ensemble_init_func = import_callable(ensemble_init_func)
        models = ensemble_init_func(pipe, ensemble_size=ensemble_size)
    if model_selection:
        model_selection = import_callable(model_selection)
    fit_score_kwargs = {}
    fit_score_kwargs['scoring'] = scoring
    fit_score_kwargs['scoring_kwargs'] = scoring_kwargs or {}
    fit_score_kwargs['method'] = method
    fit_score_kwargs['partial_fit_batches'] = partial_fit_batches or 1
    fit_score_kwargs['classes'] = classes
    final_names = []
    if X is None:
        dsk = make_samples(pipe, args_gen, sampler)
    else:
        dsk = {_next_name(): (lambda: X, y, sample_weight,)}
    models = tuple(zip(('tag_{}'.format(idx) for idx in range(len(models))), models))
    for gen in range(ngen):
        logger.info('Ensemble generation {} of {} - len(models): {} '.format(gen + 1, ngen, len(models)))
        models = _run_fit_and_score_dask(dsk,
                                         models,
                                         fit_score_kwargs,
                                         get_func,
                                         models_share_sample,
                                         method=method,
                                         sample_key=None)
        if model_selection:
            models = _run_model_selection(model_selection,
                                               model_args,
                                               ngen,
                                               gen,
                                               fit_kwargs,
                                               models)
        else:
            pass # just training all ensemble members
                 # without replacing / re-ininializing / editing
                 # the model params
    if serialize_models:
        _serialize_models(models, **ensemble_kwargs)
    return models


