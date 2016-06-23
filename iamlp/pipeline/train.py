import copy
from functools import partial

from iamlp.pipeline.sample_util import all_sample_ops
from iamlp.model_selectors.util import get_args_kwargs_defaults
from iamlp.config import import_callable
from iamlp.pipeline.ensemble import ensemble
def no_selector(models, *args, **kwargs):
    return models

def train_step(config, step, executor):
    SERIAL_EVAL = config.SERIAL_EVAL
    train_dict = config.train[step['train']]
    sample_meta = all_sample_ops(train_dict, config, step)
    action_data, sampler, sampler_args, data_source, included_filenames = sample_meta
    model_init_func = import_callable(train_dict['model_init_func'])
    _, model_init_kwargs = get_args_kwargs_defaults(model_init_func)
    model_init_kwargs.update(train_dict['model_init_kwargs'])
    if 'batch_size' in model_init_kwargs:
        model_init_kwargs['batch_size'] = sampler['n_rows_per_sample']
    post_fit_func = train_dict.get('post_fit_func', None)
    partial_fit_args = (action_data,)
    fit_kwargs = train_dict['fit_kwargs']
    fit_kwargs = {
        'post_fit_func': post_fit_func,
        'selection_kwargs': sampler.get('selection_kwargs') or {},
    }
    ensemble_kwargs = train_dict['ensemble_kwargs']
    model_selector_kwargs = copy.deepcopy(train_dict['model_selector_kwargs'])
    model_selector_kwargs.update({
        'model_init_func': model_init_func,
        'model_init_kwargs': model_init_kwargs,
    })
    model_selector_func = train_dict.get('model_selector_func') or None
    if not model_selector_func:
        model_selector_func = 'iamlp.pipeline.train:no_selector'
    fit_func = import_callable(train_dict['fit_func'],
                                                   True,
                                                   train_dict['fit_func'])
    models = ensemble(executor,
             model_init_func,
             model_init_kwargs,
             fit_func,
             partial_fit_args,
             fit_kwargs,
             model_selector_func,
             model_selector_kwargs,
             **ensemble_kwargs)
    return models