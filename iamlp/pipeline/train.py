from functools import partial

from iamlp.pipeline.sample_util import all_sample_ops

def no_selector(models, *args, **kwargs):
    return models

def train_step(config, step, executor):
    SERIAL_EVAL = config['SERIAL_EVAL']
    train_dict = config.train[step['train']]
    sample_meta = all_sample_ops(train_dict, config)
    action_data, sampler, data_source, included_filenames = sample_meta
    model_init_func = train_dict['model_init_func']
    _, model_init_kwargs = get_args_kwargs_defaults(model_init_func)
    model_init_kwargs.update(train_dict['model_init_kwargs'])
    if 'batch_size' in model_init_kwargs:
        model_init_kwargs['batch_size'] = sampler['n_rows_per_sample']
    post_fit_func = sampler.get('post_fit_func', {}).get('callable')
    partial_fit_args = (sampler['n_batches'],
                        sampler_func,
                        sampler.get('selection_kwargs', {}) or {})
    partial_fit_kwargs = {
        'on_each_sample': make_on_each_sample_func(config, step),
        'post_fit_func': train_dict.get('post_fit_func', {}).get('callable', None),
    }
    ensemble_kwargs = train_dict['ensemble_kwargs']
    ensemble_kwargs['partial_fit_kwargs'] = partial_fit_kwargs
    model_selector_kwargs = copy.deepcopy(train_dict['model_selector_kwargs'])
    model_selector_kwargs.update({
        'model_init_func': model_init_func,
        'model_init_kwargs': model_init_kwargs,
    })
    model_selector_func = train_dict.get('model_selector_func', {}).get('callable', None)
    if model_selector_func is not None:
        model_selector_kwargs = copy.deecopy(train_dict['model_selector_kwargs'])
        model_selector_kwargs['model_init_kwargs'] = model_init_kwargs
    else:
        model_selector_func_partial = 'iamlp.pipeline.train:no_selector'
    models = ensemble(model_init_func,
                      model_init_kwargs,
                      partial_fit_func,
                      partial_fit_args,
                      partial_fit_kwargs,
                      model_selector_func,
                      model_selector_kwargs,
                      **ensemble_kwargs)
    return models