from collections import namedtuple, Sequence
import copy
import inspect
import logging

import numbers

from elm.config import import_callable
from elm.model_selection.base import (base_selection, no_selection)
from elm.model_selection.scoring import score_one_model
from elm.model_selection.sorting import pareto_front
from elm.model_selection.util import ModelArgs
from elm.pipeline.fit import fit
from elm.pipeline.sample_pipeline import get_sample_pipeline_action_data
from elm.pipeline.serialize import save_models_with_meta

logger = logging.getLogger(__name__)

def _make_model_args_from_config(config, train_or_trans_dict,
                                 step, train_or_transform):
    from elm.config.util import import_callable
    from elm.model_selection.util import get_args_kwargs_defaults
    action_data = get_sample_pipeline_action_data(train_or_trans_dict, config, step)
    data_source = config.data_sources[train_or_trans_dict.get('data_source')]
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
    fit_method = step.get('method', train_or_trans_dict.get('fit_method', train_or_trans_dict.get('method', 'fit')))
    if 'batch_size' in model_init_kwargs:
        batch_size = step.get('batch_size', train_or_trans_dict.get('batch_size', data_source.get('batch_size')))
        if (not isinstance(batch_size, numbers.Number) or batch_size <= 0) and 'partial_fit' == fit_method:
            raise ValueError('"batch_size" (int) must be given in pipeline train or '
                             'transform when partial_fit is used as a fit_method')
        if fit_method == 'partial_fit':
            model_init_kwargs['batch_size'] = batch_size
    fit_args = (action_data,)
    fit_kwargs = {
        'get_y_func': data_source.get('get_y_func'),
        'get_y_kwargs': data_source.get('get_y_kwargs'),
        'get_weight_func': data_source.get('get_weight_func'),
        'get_weight_kwargs': data_source.get('get_weight_kwargs'),
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
                           fit_method,
                           fit_args,
                           fit_kwargs,
                           model_scoring,
                           model_scoring_kwargs,
                           model_selection_func,
                           model_selection_kwargs,
                           train_or_transform,
                           step[train_or_transform],
                    )
    return model_args


def make_model_args_from_config(config,
                                step,
                                train_or_transform):
    if train_or_transform == 'train':
        train_or_trans_dict = config.train[step['train']]
        train_model_args = _make_model_args_from_config(config,
                                                        train_or_trans_dict,
                                                        step,
                                                        train_or_transform)
        transform_model_args = None
    else:
        train_model_args = None
        train_or_trans_dict = config.transform[step['transform']]
        transform_model_args = _make_model_args_from_config(config,
                                                            train_or_trans_dict,
                                                            step,
                                                            train_or_transform)
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


def _prepare_fit_kwargs(model_args, transform_model, ensemble_kwargs):
    fit_kwargs = copy.deepcopy(model_args.fit_kwargs)
    fit_kwargs['scoring'] = model_args.model_scoring
    fit_kwargs['scoring_kwargs'] = model_args.model_scoring_kwargs
    fit_kwargs['transform_model'] = transform_model
    fit_kwargs['fit_method'] = model_args.fit_method
    fit_kwargs['batches_per_gen'] = ensemble_kwargs.get('batches_per_gen') or 1
    return fit_kwargs


def _get_model_selection_func(model_args):
    if model_args.model_selection_func and model_args.model_selection_func != 'no_selection':
        model_selection_func = import_callable(model_args.model_selection_func)
    else:
        model_selection_func = no_selection
    return model_selection_func


def _run_model_selection_func(model_selection_func, model_args,
                              n_generations, generation,
                              fit_kwargs, models):
    model_selection_kwargs = copy.deepcopy(model_args.model_selection_kwargs)
    model_selection_kwargs['n_generations'] = n_generations
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


def serialize_models(models, **ensemble_kwargs):
    if ensemble_kwargs.get('saved_ensemble_size') is not None:
        saved_models = models[:ensemble_kwargs['saved_ensemble_size']]
    else:
        saved_models = models
    model_paths, meta_path = save_models_with_meta(saved_models,
                                 ensemble_kwargs['base_output_dir'],
                                 ensemble_kwargs['tag'],
                                 ensemble_kwargs['config'])
    logger.info('Created model pickles: {} '
                'and meta pickle {}'.format(model_paths, meta_path))
    return models


def _fit_list_of_models(args_kwargs, map_function,
                        get_results, model_names):
    fitted = get_results(
                map_function(
                    lambda x: fit(*x[0], **x[1]),
                    args_kwargs
                ))
    models = tuple(zip(model_names, fitted))
    return models

