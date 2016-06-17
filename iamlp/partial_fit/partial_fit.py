import numpy as np

from iamlp.samplers import random_image_selection_gen
from iamlp.settings import delayed

@delayed
def partial_fit(model,
              filenames_gen,
              band_specs,
              n_samples_to_partial_fit=100,
              n_per_file=100000,
              files_per_sample=10,
              data_gen=None,
              **selection_kwargs):
    if data_gen is None:
        args = (filenames_gen, n_samples_to_partial_fit,
                n_per_file, files_per_sample, band_specs)
        gen = random_image_selection_gen(*args, **selection_kwargs)
    else:
        gen = data_gen()
    for idx, sample in enumerate(gen):
        df, band_metas, filemetas = sample
        model.partial_fit(df.values)
    return model, df


