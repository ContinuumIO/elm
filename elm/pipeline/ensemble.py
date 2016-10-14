from functools import partial, wraps
from itertools import chain, product
import copy
import logging
import numbers
import random

import dask

from elm.config import import_callable
from elm.config.dask_settings import _find_get_func_for_client
from elm.pipeline.fit_and_score import fit_and_score
from elm.pipeline.serialize import (load_models_from_tag,
                                    serialize_models as _serialize_models)
from elm.pipeline.util import (_validate_ensemble_members,
                               _run_model_selection,
                               _next_name)
from elm.sample_util.samplers import make_samples_dask
from elm.pipeline.fit_and_score import fit_and_score

logger = logging.getLogger(__name__)

__all__ = ['ensemble']


def _fit_once(method, model, fit_score_kwargs, args):
    from elm.pipeline import Pipeline
    assert isinstance(model, Pipeline), repr(model)
    X, y, sample_weight = args
    fitting_func = getattr(model, method, None)
    if fitting_func is None:
        raise ValueError("Estimator {} has no method {}".format(model, fitting_func))
    kw = dict(**fit_score_kwargs)
    if y is not None:
        kw['y'] = y
    if sample_weight is not None:
        kw['sample_weight'] = sample_weight
    return fitting_func(X, **kw)


def _run_fit_and_score_dask(dsk,
                            models,
                            fit_score_kwargs,
                            get_func,
                            models_share_sample,
                            sample_keys,
                            method='fit',
                            sample_key=None):

    # Samples before Pipeline actions applied
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
    models = tuple(zip(model_keys, new_models))
    return models


def ensemble(pipe,
             ngen,
             X=None,
             y=None,
             sample_weight=None,
             sampler=None,
             args_list=None,
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
    get_func = _find_get_func_for_client(client)
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
    if 'partial_fit' in method:
        method = 'partial_fit_n_batches'
        fit_score_kwargs['partial_fit_batches'] = partial_fit_batches or 1
    if classes is not None:
        fit_score_kwargs['classes'] = classes
    final_names = []
    dsk = make_samples_dask(X, y, sample_weight, pipe, args_list, sampler)
    models = tuple(zip(('tag_{}'.format(idx) for idx in range(len(models))), models))
    sample_keys = tuple(dsk)

    for gen in range(ngen):
        logger.info('Ensemble generation {} of {} - len(models): {} '.format(gen + 1, ngen, len(models)))
        models = _run_fit_and_score_dask(dsk,
                                         models,
                                         fit_score_kwargs,
                                         get_func,
                                         models_share_sample,
                                         sample_keys=sample_keys,
                                         method=method,
                                         sample_key=None)
        if model_selection:
            models = _run_model_selection(models,
                                          model_selection,
                                          model_selection_kwargs or {},
                                          ngen,
                                          gen,
                                          scoring_kwargs)

        else:
            pass # just training all ensemble members
                 # without replacing / re-ininializing / editing
                 # the model params
    if serialize_models:
        _serialize_models(models, **ensemble_kwargs)
    return models


