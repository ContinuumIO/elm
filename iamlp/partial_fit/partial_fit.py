import numpy as np

from iamlp.samplers import random_images_selection
from iamlp.settings import delayed

@delayed
def partial_fit(model,
              included_files,
              band_specs,
              n_samples_each_fit=100,
              n_per_file=100000,
              files_per_sample=10,
              data_func=None,
              post_fit_func=None,
              **selection_kwargs):
    if data_func is None:
        args = (included_files, n_samples_each_fit,
                n_per_file, files_per_sample, band_specs)
        sample = lambda: random_images_selection(*args, **selection_kwargs)
    else:
        sample = lambda: data_func()
    for idx in range(n_samples_each_fit):
        df, band_metas, filemetas = sample()
        model.partial_fit(df.values)
        if post_fit_func is not None:
            post_fit_func(model, df)
    return (model, df)


