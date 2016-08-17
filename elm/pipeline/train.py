import copy
from functools import partial
import logging
import numbers

from elm.pipeline.sample_pipeline import get_sample_pipeline_action_data
from elm.model_selection.util import get_args_kwargs_defaults
from elm.config import import_callable
from elm.pipeline.ensemble import ensemble
from elm.pipeline.serialize import load_models_from_tag
from elm.pipeline.transform import init_sample_pipeline_transform_models
logger = logging.getLogger(__name__)



def _make_model_args_from_config(config, train_dict, step):
    action_data = get_sample_pipeline_action_data(train_dict, config, step)
    data_source = config.data_sources[train_dict.get('data_source')]
    if train_dict.get('model_scoring'):
        logger.debug('Has model_scoring {}'.format(train_dict['model_scoring']))
        ms = config.model_scoring[train_dict['model_scoring']]
        model_scoring = ms['scoring']
        model_scoring_kwargs = ms
    else:
        logger.debug('No model_scoring {}'.format(train_dict))
        model_scoring = None
        model_scoring_kwargs = {}
    model_init_class = import_callable(train_dict['model_init_class'])
    _, default_model_init_kwargs, _ = get_args_kwargs_defaults(model_init_class)
    default_fit = 'partial_fit' if 'partial_fit' in dir(model_init_class) else 'fit'
    fit_method = step.get('method', train_dict.get('fit_method', train_dict.get('method', 'fit')))
    if 'batch_size' in model_init_kwargs:
        batch_size = step.get('batch_size', train_dict.get('batch_size', data_source.get('batch_size')))
        if (not isinstance(batch_size, numbers.Number) or batch_size <= 0) and 'partial_fit' == fit_method:
            raise ValueError('"batch_size" (int) must be given in pipeline train or '
                             'transform when partial_fit is used as a fit_method')
        if fit_method == 'partial_fit':
            model_init_kwargs['batch_size'] = batch_size
    default_model_init_kwargs.update(train_dict['model_init_kwargs'])
    model_init_kwargs = default_model_init_kwargs
    fit_args = (action_data,)
    fit_kwargs = {
        'get_y_func': data_source.get('get_y_func'),
        'get_y_kwargs': data_source.get('get_y_kwargs'),
        'get_weight_func': data_source.get('get_weight_func'),
        'get_weight_kwargs': data_source.get('get_weight_kwargs'),
        'fit_kwargs': train_dict.get('fit_kwargs') or {},
    }
    model_selection = train_dict.get('model_selection') or None
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
                    )
    return model_args

def _make_transform_model_args_from_config(config, step):
    sample_pipeline = step['sample_pipeline']
    for step in sample_pipeline:
        if 'transform' in step:
            method = step.get('method', config.transform.get('method', config.transform.get('fit_method', 'fit_transform')))
            if 'fit' in method:
                return _make_model_args_from_config(config, config.transform[step['transform']], step)
    return None

def _train_or_transform_step(train_or_transform, config,
                             step, executor, evo_params=None,
                             **kwargs):
    '''Evaluate a "train" step in a config's "pipeline"

    Params:
        config:  full config
        step:    current step dictionary in config's pipeline,
                 with a "train" action
        executor: None or a threaded/process/distributed Executor
    Returns:
        models: the fitted models in the ensemble method
    '''

    ensemble_kwargs = config.ensembles[train_dict['ensemble']]
    if train_or_transform in config.train:
        train_or_transform_dict = config.train[train_or_transform]
    else:
        train_or_transform_dict = config.transform[train_or_transform]
    model_args = _make_model_args_from_config(config, train_or_transform_dict, step)
    model_args.fit_kwargs['batches_per_gen'] = ensemble_kwargs.get('batches_per_gen')
    if train_or_transform == 'train':
        transform_model_args = _make_transform_model_args_from_config(config, step)
    else:
        transform_model_args = None
    if train_or_transform == 'transform':
        ensemble_kwargs['base_output_dir'] = config.ELM_TRANSFORM_PATH
    else:
        ensemble_kwargs['base_output_dir'] = config.ELM_TRAIN_PATH
    ensemble_kwargs['config'] = config
    ensemble_kwargs['tag'] = step[train_or_transform]
    if any('transform' in item for item in (step.get('sample_pipeline') or [])):
        if transform_model_args is None:
            transform_dict = kwargs.get('transform_dict') or None
            if transform_dict is None:
                transform_dict = init_sample_pipeline_transform_models(config, step)
    models = ensemble(executor,
                      model_args,
                      transform_model_args,
                      transform_dict,
                      evo_params,
                      **ensemble_kwargs)
    return models

train_step = partial(_train_or_transform_step, 'train')

