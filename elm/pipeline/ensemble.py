from functools import partial, wraps
from itertools import chain
import copy
import logging
import numbers

from elm.config import import_callable
from elm.config import import_callable
from elm.config.dask_settings import wait_for_futures
from elm.pipeline.fit_and_score import fit_and_score
from elm.pipeline.serialize import (load_models_from_tag, serialize_models)
from elm.pipeline.util import (make_model_args_from_config,
                               _prepare_fit_kwargs,
                               _validate_ensemble_members,
                               _get_model_selection_func,
                               _run_model_selection_func,
                               _ensemble_ea_prep,
                               _next_name)
from elm.sample_util.samplers import make_one_sample

logger = logging.getLogger(__name__)

__all__ = ['ensemble']


def _run_fit_and_score_dask(config, sample_pipeline, data_source,
                   transform_model, samples_per_batch, models,
                   gen, fit_kwargs, sample_pipeline_kwargs=None,
                   get_func=None, sample=None):

    train_dsk = {}
    sample_name = 'sample-{}'.format(gen)
    sample_args = (config, sample_pipeline, data_source,
                   transform_model,
                   samples_per_batch,
                   sample_name,
                   sample_pipeline_kwargs,
                   sample)
    train_dsk.update(make_one_sample(*sample_args))
    keys = tuple(name for name, model in models)
    for idx, (name, model) in enumerate(models):
        if isinstance(fit_kwargs, (tuple, list)):
            kw = fit_kwargs[idx]
        else:
            kw = fit_kwargs
        fitter = partial(fit_and_score, model, **kw)
        train_dsk[name] = (fitter, sample_name)
    def tuple_of_args(*args):
        return tuple(args)
    new_models_name = _next_name()
    train_dsk[new_models_name] = (tuple_of_args, *keys)
    if get_func is None:
        new_models = tuple(dask.get(train_dsk, new_models_name))
    else:
        new_models = tuple(get_func(train_dsk, new_models_name))
    models = tuple(zip(keys, new_models))
    return models


def ensemble(*args, **kwargs):

    '''Train model(s) in ensemble

    '''
    from elm.config.dask_settings import get_func
    args, kwargs = _ensemble_ea_prep(*args, **kwargs)
    (client,
     model_args,
     transform_model,
     sample_pipeline,
     data_source,) = args
    samples_per_batch = kwargs.get('samples_per_batch', 1)
    config = kwargs['config']
    sample_pipeline_kwargs = kwargs['sample_pipeline_kwargs']
    sample = kwargs['sample']
    ensemble_kwargs = kwargs.get('ensemble_kwargs')
    if hasattr(client, 'map'):
        map_function = client.map
    else:
        map_function = map
    n_batches = data_source.get('n_batches') or 1
    get_results = partial(wait_for_futures, client=client)
    model_selection_kwargs = model_args.model_selection_kwargs or {}
    ensemble_size = ensemble_kwargs.get('init_ensemble_size', None)
    if not ensemble_size:
        logger.info('Setting ensemble_kwargs["init_ensemble_size"] = 1')
        ensemble_kwargs['init_ensemble_size'] = ensemble_size = 1
    ngen = ensemble_kwargs['ngen']
    ensemble_init_func = ensemble_kwargs.get('ensemble_init_func') or None
    model_init_kwargs = model_args.model_init_kwargs
    model_init_class = model_args.model_init_class
    if not ensemble_init_func:
        models = tuple(model_init_class(**model_init_kwargs)
                       for idx in range(ensemble_size))
    else:
        ensemble_init_func = import_callable(ensemble_init_func)
        models = ensemble_init_func(model_init_class,
                                    model_init_kwargs,
                                    **ensemble_kwargs)
    model_selection_func = _get_model_selection_func(model_args)
    fit_kwargs = _prepare_fit_kwargs(model_args, ensemble_kwargs)
    final_names = []
    models = tuple(zip(('tag_{}'.format(idx) for idx in range(len(models))), models))
    for gen in range(ngen):
        logger.info('Ensemble generation {} of {} - len(models): {} '.format(gen + 1, ngen, len(models)))
        models = _run_fit_and_score_dask(config, sample_pipeline, data_source,
                                transform_model,
                                samples_per_batch, models,
                                gen, fit_kwargs,
                                sample_pipeline_kwargs=sample_pipeline_kwargs,
                                get_func=get_func,
                                sample=sample)
        if model_selection_func:
            models = _run_model_selection_func(model_selection_func,
                                               model_args,
                                               ngen,
                                               gen,
                                               fit_kwargs,
                                               models)
        else:
            pass # just training all ensemble members
                 # without replacing / re-ininializing / editing
                 # the model params
    serialize_models(models, **ensemble_kwargs)
    return models


