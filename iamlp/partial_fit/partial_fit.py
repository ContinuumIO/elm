import copy
import numpy as np

from iamlp.samplers import random_images_selection
from iamlp.config.settings import delayed, SERIAL_EVAL

def partial_fit(model,
                n_batches,
                sampler_func,
                selection_kwargs,
                on_each_sample=None,
                post_fit_func=None,
                **on_each_sample_kwargs):
    selection_kwargs = kwargs.get('selection_kwargs') or {}
    selection_kwargs = copy.deepcopy(selection_kwargs)
    n_batches = kwargs['n_batches']
    for idx in range(n_batches):
        sample = sampler_func(**selection_kwargs)
        if on_each_sample is not None:
            sample = on_each_sample(sample, **on_each_sample_kwargs)
        model = model.partial_fit(sample.df.values)
    if post_fit_func is not None:
        return post_fit_func(model, sample.df)
    return model


