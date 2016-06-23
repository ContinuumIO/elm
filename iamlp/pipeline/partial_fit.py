import copy
import numpy as np

from iamlp.samplers import random_images_selection
from iamlp.config.dask_settings import delayed, SERIAL_EVAL
from iamlp.pipeline.sample_util import run_sample_pipeline
from iamlp.config import import_callable
FIT_FUNC_ERR = ('Expected model {} to have a '
                '"partial_fit" method with n_batches = {}.\n'
                'Set n_batches to 1 or use model with "partial_fit" method')
def partial_fit(model,
                action_data,
                post_fit_func=None,
                n_batches=2,
                **on_each_sample_kwargs):
    if post_fit_func is not None:
        pff = import_callable(post_fit_func, True, post_fit_func)
    selection_kwargs = on_each_sample_kwargs.get('selection_kwargs') or {}
    selection_kwargs = copy.deepcopy(selection_kwargs)
    if n_batches > 1:
        if not hasattr(model, 'partial_fit'):
            raise ValueError(FIT_FUNC_ERR.format(repr(model), n_batches))

    for idx in range(n_batches):
        sample = run_sample_pipeline(action_data, **on_each_sample_kwargs)
        model = model.partial_fit(sample.df.values)
    if post_fit_func is not None:
        return pff(model, sample.df)
    return model


