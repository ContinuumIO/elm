import copy
from functools import partial

from elm.pipeline.sample_pipeline import all_sample_ops
from elm.model_selection.util import get_args_kwargs_defaults
from elm.config import import_callable
from elm.pipeline.ensemble import ensemble
def no_selection(models, *args, **kwargs):
    return models

def train_step(config, step, executor):
    SERIAL_EVAL = config.SERIAL_EVAL
    train_dict = config.train[step['train']]
    sample_meta = all_sample_ops(train_dict, config, step)
    action_data = sample_meta[:2]
    sampler = train_dict.get('sampler')
    if sampler:
        sampler = config.samplers[sampler]
    else:
        sampler = config.samplers[train_dict['data_source']]
    model_init_class = import_callable(train_dict['model_init_class'])
    _, model_init_kwargs = get_args_kwargs_defaults(model_init_class)
    model_init_kwargs.update(train_dict['model_init_kwargs'])
    if 'batch_size' in model_init_kwargs:
        model_init_kwargs['batch_size'] = sampler['n_rows_per_sample']
    post_fit_func = train_dict.get('post_fit_func', None)
    fit_args = (action_data,)
    fit_kwargs = train_dict['fit_kwargs']
    fit_kwargs = {
        'post_fit_func': post_fit_func,
        'selection_kwargs': sampler.get('selection_kwargs') or {},
    }
    ensemble_kwargs = train_dict['ensemble_kwargs']
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