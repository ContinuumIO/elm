import copy
import numpy as np

from iamlp.samplers import random_images_selection
from iamlp.config.settings import delayed, SERIAL_EVAL

@delayed
def partial_fit(model,
                sampler_func=None,
                on_each_sample=None,
                n_batches=3,
                post_fit_func=None,
                selection_kwargs=None):
    selection_kwargs = selection_kwargs or {}
    selection_kwargs = copy.deepcopy(selection_kwargs)
    for idx in range(n_batches):
        kwargs = selection_kwargs or {}
        kwargs['iteration'] = idx
        sample = sampler_func(**kwargs)
        sample = on_each_sample(sample)
        if not SERIAL_EVAL:
            sample = sample.compute()  #TODO: should .compute be here?
                                       # if so, the get= argument
                                       # needs to be given.
        model = delayed(model.partial_fit)(sample.df.values)
    if post_fit_func is not None:
        return post_fit_func(model, sample.df)
    return model


