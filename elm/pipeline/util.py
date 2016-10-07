from collections import namedtuple, Sequence
import copy
from functools import partial
import inspect
import logging

import dask
import numbers

from elm.model_selection.util import get_args_kwargs_defaults, ModelArgs
from elm.model_selection.evolve import ea_setup
from elm.config import import_callable, parse_env_vars
from elm.model_selection.base import (base_selection, no_selection)
from elm.model_selection.scoring import score_one_model
from elm.model_selection.sorting import pareto_front
from elm.model_selection.util import ModelArgs
from elm.sample_util.sample_pipeline import get_sample_pipeline_action_data
from elm.sample_util.samplers import make_one_sample



logger = logging.getLogger(__name__)

NO_ENSEMBLE = {'init_ensemble_size': 1,
               'ngen': 1,
               'partial_fit_batches': 1,
               'saved_ensemble_size': 1,}

_next_idx = 0

def _next_name():
    global _next_idx
    n = 'ensemble-{}'.format(_next_idx)
    _next_idx += 1
    return n

def _make_model_args_from_config(config, train_or_trans_dict,
                                 step, train_or_transform,
                                 sample_pipeline, data_source,
                                 model_scoring=None,
                                 model_selection=None,
                                 **sample_pipeline_kwargs):
    from elm.config.util import import_callable
    from elm.model_selection.util import get_args_kwargs_defaults
    classes = train_or_trans_dict.get('classes', None)

    action_data = get_sample_pipeline_action_data(sample_pipeline, config, step,
                                                  data_source, **sample_pipeline_kwargs)
    if not model_scoring:
        model_scoring = train_or_trans_dict.get('model_scoring')
    if model_scoring:
        logger.debug('Has model_scoring {}'.format(model_scoring))
        if config and not isinstance(model_scoring, dict):
            ms = config.model_scoring[train_or_trans_dict['model_scoring']]
        else:
            ms = model_scoring
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
    method = train_or_trans_dict.get('method', None)
    if not method and step:
        method = step.get('method')
    if not method:
        method = 'fit'
    if 'batch_size' in model_init_kwargs:
        batch_size = train_or_trans_dict.get('batch_size', data_source.get('batch_size'))
        if not batch_size and step:
            batch_size = step.get('batch_size')
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
    if model_selection:
        if isinstance(model_selection, str):
            if model_selection == 'no_selection':
                model_selection = None
            else:
                model_selection = config.model_selection[model_selection]
        if model_selection:
            model_selection_kwargs = copy.deepcopy(model_selection.get('kwargs') or {}) or {}
            model_selection_kwargs.update({
                'model_init_class': model_init_class,
                'model_init_kwargs': model_init_kwargs,
            })
            model_selection_func = model_selection['func']
        else:
            model_selection_func = 'no_selection'
            model_selection_kwargs = {}
    else:
        model_selection_func = 'no_selection'
        model_selection_kwargs = {}
    step_name = step[train_or_transform] if step else train_or_trans_dict['output_tag']
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
                           step_name,
                           classes)
    return model_args


def make_model_args_from_config(train_or_transform,
                                sample_pipeline,
                                data_source,
                                config=None,
                                step=None,
                                train_dict=None,
                                model_scoring=None,
                                model_selection=None,
                                ensemble_kwargs=None,
                                **sample_pipeline_kwargs):
    if train_or_transform == 'train':
        if config and step:
            train_or_trans_dict = config.train[step['train']]
        else:
            train_or_trans_dict = train_dict
        train_model_args = _make_model_args_from_config(config,
                                                        train_or_trans_dict,
                                                        step,
                                                        train_or_transform,
                                                        sample_pipeline,
                                                        data_source,
                                                        model_scoring=model_scoring,
                                                        model_selection=model_selection,
                                                        **sample_pipeline_kwargs)
        transform_model_args = None
    else:

        train_model_args = None
        if config and step:
            train_or_trans_dict = config.transform[step['transform']]
        else:
            train_or_trans_dict = train_dict or sample_pipeline_kwargs.get('transform_dict') or {}
        transform_model_args = _make_model_args_from_config(config,
                                                            train_or_trans_dict,
                                                            step,
                                                            train_or_transform,
                                                            sample_pipeline,
                                                            data_source,
                                                            None,
                                                            **sample_pipeline_kwargs)
    if not isinstance(ensemble_kwargs, dict):
        ensemble_kwargs = config.ensembles[train_or_trans_dict['ensemble']]
        ensemble_kwargs['tag'] = (train_model_args or transform_model_args).step_name
    if not config:
        env = parse_env_vars()
        if train_or_transform == 'transform':
            ensemble_kwargs['base_output_dir'] = env['ELM_TRANSFORM_PATH']
        else:
            ensemble_kwargs['base_output_dir'] = env['ELM_TRAIN_PATH']
    else:
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
    if not any(isinstance(m, Sequence) for m in models):
        # list of models with no tags - make some up
        return [(_next_name(), m) for m in models]
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


def _ensemble_ea_prep(sample_pipeline,
                      data_source,
                      config=None,
                      step=None,
                      train_dict=None,
                      transform_model=None,
                      transform_dict=None,
                      client=None,
                      model_args=None,
                      ensemble_kwargs=None,
                      evo_params=None,
                      evo_dict=None,
                      samples_per_batch=1,
                      sample=None,
                      train_or_transform='train',
                      **sample_pipeline_kwargs):
    '''
    '''
    from elm.pipeline.transform import get_new_or_saved_transform_model
    ensemble_kwargs = ensemble_kwargs or NO_ENSEMBLE.copy()
    if not (train_dict or transform_dict) and (not config or not step):
        raise ValueError("Expected 'train_dict' and 'transform_dict' or 'config' and 'step'")
    evo_params = evo_params or None
    if model_args and evo_params:
        raise ValueError('Do not pass "model_args" when using "evo_params"')
    if evo_dict:
        if evo_params:
            raise ValueError('Cannot give evo_params and evo_dict')
        config, step = _make_config( train_or_transform,
                                     sample_pipeline, data_source,
                                     samples_per_batch, train_dict,
                                     transform_dict, ensemble_kwargs,
                                     evo_dict, **sample_pipeline_kwargs)
        evo_params_dict = ea_setup(config)
        evo_params = tuple(evo_params_dict.values())[0]

    transform_model = transform_model or None
    sample_pipeline_kwargs['transform_dict'] = transform_dict
    sample_pipeline_kwargs['transform_model'] = transform_model
    if not ensemble_kwargs and config and step:
        t = getattr(config, train_or_transform)[step[train_or_transform]]
        e = t.get('ensemble')
        if isinstance(e, dict):
            ensemble_kwargs = e
        else:
            ensemble_kwargs = config.ensembles[e]
    model_args2, ensemble_kwargs2 = make_model_args_from_config(train_or_transform,
                                                                sample_pipeline,
                                                                data_source,
                                                                config=config,
                                                                step=step,
                                                                train_dict=train_dict,
                                                                ensemble_kwargs=ensemble_kwargs,
                                                                **sample_pipeline_kwargs)
    if not model_args:
        model_args = model_args2
    ensemble_kwargs.update(ensemble_kwargs2)
    if transform_model is None and config and step:
        transform_model = get_new_or_saved_transform_model(config,
                                                           sample_pipeline,
                                                           data_source,
                                                           step)

    if not ensemble_kwargs.get('tag'):
        ensemble_kwargs['tag'] = model_args.step_name

    if evo_params or evo_dict:
        args = (train_or_transform,
                client,
                step,
                evo_params,
                evo_dict,
                config,
                sample_pipeline,
                data_source,
                transform_model,
                samples_per_batch,
                train_dict,
                transform_dict,
                sample_pipeline_kwargs,
                sample)
        return (args, ensemble_kwargs)
    args = (client,
            model_args,
            transform_model,
            sample_pipeline,
            data_source,)
    kwargs = dict(samples_per_batch=samples_per_batch,
                  config=config,
                  sample_pipeline_kwargs=sample_pipeline_kwargs,
                  sample=sample,
                  ensemble_kwargs=ensemble_kwargs)
    return (args, kwargs)


