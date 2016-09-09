from collections import namedtuple, Sequence
import copy
from functools import partial
import inspect
import logging

import dask
import numbers

from elm.config import import_callable
from elm.model_selection.base import (base_selection, no_selection)
from elm.model_selection.scoring import score_one_model
from elm.model_selection.sorting import pareto_front
from elm.model_selection.util import ModelArgs
from elm.pipeline.run_model_method import run_model_method
from elm.sample_util.sample_pipeline import get_sample_pipeline_action_data
from elm.sample_util.samplers import make_one_sample


logger = logging.getLogger(__name__)

_next_idx = 0

def _next_name():
    global _next_idx
    n = 'ensemble-{}'.format(_next_idx)
    _next_idx += 1
    return n

def _make_model_args_from_config(config, train_or_trans_dict,
                                 step, train_or_transform,
                                 sample_pipeline, data_source):
    from elm.config.util import import_callable
    from elm.model_selection.util import get_args_kwargs_defaults
    classes = train_or_trans_dict.get('classes', None)
    action_data = get_sample_pipeline_action_data(config, step,
                                    data_source, sample_pipeline)
    if train_or_trans_dict.get('model_scoring'):
        logger.debug('Has model_scoring {}'.format(train_or_trans_dict['model_scoring']))
        ms = config.model_scoring[train_or_trans_dict['model_scoring']]
        model_scoring = ms.get('scoring') or None
        model_scoring_kwargs = ms or {}
    else:
        logger.debug('No model_scoring {}'.format(train_or_trans_dict))
        model_scoring = None
        model_scoring_kwargs = {}
    model_init_class = import_callable(train_or_trans_dict['model_init_class'])
    _, model_init_kwargs, _ = get_args_kwargs_defaults(model_init_class)
    default_fit = 'partial_fit' if 'partial_fit' in dir(model_init_class) else 'fit'
    model_init_kwargs.update(train_or_trans_dict.get('model_init_kwargs') or {})
    method = step.get('method', train_or_trans_dict.get('method', 'fit'))
    if 'batch_size' in model_init_kwargs:
        batch_size = step.get('batch_size', train_or_trans_dict.get('batch_size', data_source.get('batch_size')))
        if (not isinstance(batch_size, numbers.Number) or batch_size <= 0) and 'partial_fit' == method:
            raise ValueError('"batch_size" (int) must be given in pipeline train or '
                             'transform when partial_fit is used as a method')
        if method == 'partial_fit':
            model_init_kwargs['batch_size'] = batch_size
    fit_args = (action_data,)
    fit_kwargs = {
        'fit_kwargs': train_or_trans_dict.get('fit_kwargs') or {},
    }
    model_selection = train_or_trans_dict.get('model_selection') or None
    if model_selection and model_selection not in ('no_selection',):
        model_selection = config.model_selection[model_selection]
        model_selection_kwargs = copy.deepcopy(model_selection.get('kwargs') or {}) or {}
        model_selection_kwargs.update({
            'model_init_class': model_init_class,
            'model_init_kwargs': model_init_kwargs,
        })
        model_selection_func = model_selection['func']
    else:
        model_selection_func = 'no_selection'
        model_selection_kwargs = {}
    model_args = ModelArgs(model_init_class,
                           model_init_kwargs,
                           method,
                           fit_args,
                           fit_kwargs,
                           model_scoring,
                           model_scoring_kwargs,
                           model_selection_func,
                           model_selection_kwargs,
                           train_or_transform,
                           step[train_or_transform],
                           classes,
                    )
    return model_args


def make_model_args_from_config(config,
                                step,
                                train_or_transform,
                                sample_pipeline,
                                data_source):
    if train_or_transform == 'train':
        train_or_trans_dict = config.train[step['train']]
        train_model_args = _make_model_args_from_config(config,
                                                        train_or_trans_dict,
                                                        step,
                                                        train_or_transform,
                                                        sample_pipeline,
                                                        data_source)
        transform_model_args = None
    else:
        train_model_args = None
        train_or_trans_dict = config.transform[step['transform']]
        transform_model_args = _make_model_args_from_config(config,
                                                            train_or_trans_dict,
                                                            step,
                                                            train_or_transform,
                                                            sample_pipeline,
                                                            data_source)
    ensemble_kwargs = config.ensembles[train_or_trans_dict['ensemble']]
    ensemble_kwargs['config'] = config
    ensemble_kwargs['tag'] = step[train_or_transform]
    if train_or_transform == 'transform':
        ensemble_kwargs['base_output_dir'] = config.ELM_TRANSFORM_PATH
    else:
        ensemble_kwargs['base_output_dir'] = config.ELM_TRAIN_PATH
    return (train_model_args or transform_model_args, ensemble_kwargs)


def get_transform_name_for_sample_pipeline(step):
    for item in (step.get('sample_pipeline') or []):
        if 'transform' in item:
            return item['transform'] # TODO we need to validate
                                     # that only one transform
                                     # key is used in a sample_pipeline


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


def _prepare_fit_kwargs(model_args, ensemble_kwargs):
    fit_kwargs = copy.deepcopy(model_args.fit_kwargs)
    fit_kwargs['scoring'] = model_args.model_scoring
    fit_kwargs['scoring_kwargs'] = model_args.model_scoring_kwargs
    fit_kwargs['method'] = model_args.method
    fit_kwargs['partial_fit_batches'] = ensemble_kwargs.get('partial_fit_batches') or 1
    fit_kwargs['classes'] = model_args.classes
    return fit_kwargs


def _get_model_selection_func(model_args):
    if model_args.model_selection_func and model_args.model_selection_func != 'no_selection':
        model_selection_func = import_callable(model_args.model_selection_func)
    else:
        model_selection_func = no_selection
    return model_selection_func


def _run_model_selection_func(model_selection_func, model_args,
                              ngen, generation,
                              fit_kwargs, models):
    model_selection_kwargs = copy.deepcopy(model_args.model_selection_kwargs)
    model_selection_kwargs['ngen'] = ngen
    model_selection_kwargs['generation'] = generation
    score_weights = fit_kwargs['scoring_kwargs'].get('score_weights') or None
    sort_fitness = model_args.model_scoring_kwargs.get('sort_fitness', model_selection_kwargs.get('sort_fitness')) or None
    if not sort_fitness:
        sort_fitness = pareto_front
    else:
        sort_fitness = import_callable(sort_fitness)
    logger.debug('base_selection {}'.format(repr((models, model_args.model_selection_func, sort_fitness, score_weights, model_selection_kwargs))))
    models = base_selection(models,
                            model_selection_func=model_selection_func,
                            sort_fitness=sort_fitness,
                            score_weights=score_weights,
                            **model_selection_kwargs)
    models = _validate_ensemble_members(models)
    return models



def _fit_one_model(*fit_args, **fit_kwargs):
    return run_model_method(*fit_args, **fit_kwargs)


def _fit_list_of_models(args_kwargs, map_function,
                        get_results, model_names):

    fitted = get_results(
                map_function(
                    lambda x: fit(*x[0], **x[1]),
                    args_kwargs
                ))
    models = tuple(zip(model_names, fitted))
    return models

def run_train_dask(sample_pipeline_info, models,
                   new_models, gen, fit_kwargs,
                   get_func=None):
    train_dsk = {}
    sample_name = 'sample-{}'.format(gen)
    sample_args = tuple(sample_pipeline_info) + (sample_name,)
    train_dsk.update(make_one_sample(*sample_args))
    keys = tuple('model-{}-gen-{}'.format(idx, gen)
                 for idx in range(len(new_models or models)))
    for name, (_, model) in zip(keys, models):
        fitter = partial(_fit_one_model, model, **fit_kwargs)
        train_dsk[name] = (fitter, sample_name)
    def tkeys(*args):
        return tuple(args)
    new_models_name = _next_name()
    train_dsk[new_models_name] = (tkeys, *keys)
    if get_func is None:
        new_models = tuple(dask.get(train_dsk, new_models_name))
    else:
        new_models = tuple(get_func(train_dsk, new_models_name))
    models = tuple(zip(keys, new_models))
    return models

