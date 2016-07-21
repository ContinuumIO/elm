import copy
from functools import partial

from elm.pipeline.sample_pipeline import all_sample_ops
from elm.model_selection.util import get_args_kwargs_defaults
from elm.config import import_callable
from elm.pipeline.ensemble import ensemble
def no_selection(models, *args, **kwargs):
    return models

def train_step(config, step, executor, **kwargs):
    '''Evaluate a "train" step in a config's "pipeline"

    Params:
        config:  full config
        step:    current step dictionary in config's pipeline,
                 with a "train" action
        executor: None or a threaded/process/distributed Executor
    Returns:
        models: the fitted models in the ensemble method
    '''
    train_dict = config.train[step['train']]
    action_data = all_sample_ops(train_dict, config, step)
    data_source = train_dict.get('data_source')

    model_init_class = import_callable(train_dict['model_init_class'])
    _, model_init_kwargs = get_args_kwargs_defaults(model_init_class)
    model_init_kwargs.update(train_dict['model_init_kwargs'])
    fit_args = (action_data,)
    fit_kwargs = {
        'post_fit_func': train_dict.get('post_fit_func'),
        'fit_func': train_dict['fit_func'],
        'get_y_func': train_dict.get('get_y_func'),
        'get_y_kwargs': train_dict.get('get_y_kwargs'),
        'get_weight_func': train_dict.get('get_weight_func'),
        'get_weight_kwargs': train_dict.get('get_weight_kwargs'),
        'batches_per_gen': train_dict['ensemble_kwargs'].get('batches_per_gen'),
        'fit_kwargs': train_dict['fit_kwargs'],
    }
    ensemble_kwargs = train_dict['ensemble_kwargs']
    ensemble_kwargs['config'] = config
    ensemble_kwargs['tag'] = step['train']
    model_selection_kwargs = copy.deepcopy(train_dict['model_selection_kwargs'])
    model_selection_kwargs.update({
        'model_init_class': model_init_class,
        'model_init_kwargs': model_init_kwargs,
    })
    model_selection_func = train_dict.get('model_selection_func') or None
    if not model_selection_func:
        model_selection_func = 'elm.pipeline.train:no_selection'
    fit_func = train_dict['fit_func']
    models = ensemble(executor,
             model_init_class,
             model_init_kwargs,
             fit_func,
             fit_args,
             fit_kwargs,
             model_selection_func,
             model_selection_kwargs,
             **ensemble_kwargs)
    return models
