from functools import partial

from iamlp.pipeline.sample_util import (make_no_args_sampler_func,
                                        make_on_each_sample_func)

def no_selector(models, *args, **kwargs):
    return models

def train_step(config, step):
    SERIAL_EVAL = config['SERIAL_EVAL']
    train_dict = config.train[step['train']]
    out = make_no_args_sampler_func(train_dict, config)
    no_args_sampler, sampler, sampler_args, sampler_kwargs = out
    model_init_func = train_dict['model_init']['callable']
    _, model_init_kwargs = get_args_kwargs_defaults(model_init_func)
    model_init_kwargs.update(train_dict['model_init_kwargs'])
    if 'batch_size' in model_init_kwargs:
        model_init_kwargs['batch_size'] = sampler['n_rows_per_sample']
    post_fit_func = sampler.get('post_fit_func', {}).get('callable')
    partial_fit_kwargs = {
        'sampler_func': sampler_func,
        'on_each_sample': make_on_each_sample_func(config, step),
        'n_batches': sampler['n_batches'],
        'selection_kwargs': sampler.get('selection_kwargs', {}) or {},
        'post_fit_func': train_dict.get('post_fit_func', {}).get('callable', None),
    }
    fit_function = partial(partial_fit, **partial_fit_kwargs)
    ensemble_kwargs = train_dict['ensemble_kwargs']
    ensemble_kwargs['partial_fit_kwargs'] = partial_fit_kwargs
    model_selector_kwargs = copy.deepcopy(train_dict['model_selector_kwargs'])
    model_selector_kwargs.update({
        'model_init_func': model_init_func,
        'model_init_kwargs': model_init_kwargs,
    })
    model_selector = train_dict.get('model_selector', {}).get('callable', None)
    if model_selector is not None:
        model_selector_kwargs = copy.deecopy(train_dict['model_selector_kwargs'])
        model_selector_kwargs['model_init_kwargs'] = model_init_kwargs
        model_selector_partial = partial(model_selector, **model_selector_kwargs)
    else:
        model_selector_partial = no_selector
    models = ensemble(init_models,
                      fit_function,
                      model_selector_partial,
                      **ensemble_kwargs)
    if not SERIAL_EVAL:
        # TODO is this the right place?
        models = [m.compute() for m in models.compute()]
    return models