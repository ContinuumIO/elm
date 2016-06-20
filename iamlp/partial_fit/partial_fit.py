import numpy as np

from iamlp.samplers import random_images_selection
from iamlp.settings import delayed

@delayed
def partial_fit(model,
                band_specs=None,
                n_samples_each_fit=100,
                n_per_file=100000,
                files_per_sample=10,
                data_func=None,
                post_fit_func=None,
                included_filenames=None,
                selection_kwargs=None):
    selection_kwargs = selection_kwargs or {}
    post_fit_func_delayed = delayed(post_fit_func)
    if data_func is None:
        args = (included_filenames, n_samples_each_fit,
                n_per_file, files_per_sample, band_specs)
        assert included_filenames
        sample = lambda: random_images_selection(*args, **selection_kwargs)
    else:
        sample = lambda: data_func()
    for idx in range(n_samples_each_fit):
        samp = sample()
        delayed(model.partial_fit)(samp.df.values)
        if post_fit_func is not None:
            post_fit_func_delayed(model, samp.df)
        model.df = samp.df
    return model


