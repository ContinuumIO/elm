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
                fit_func='partial_fit',
                get_y_func=None,
                get_y_kwargs=None,
                post_fit_func=None,
                n_batches=2,
                feature_selector='all',
                fit_kwargs=None,
                **sample_pipeline_kwargs):
    if post_fit_func is not None:
        pff = import_callable(post_fit_func, True, post_fit_func)
    selection_kwargs = sample_pipeline_kwargs.get('selection_kwargs') or {}
    selection_kwargs = copy.deepcopy(selection_kwargs)
    get_y_kwargs = get_y_kwargs or {}
    if n_batches > 1:
        if not fit_func == 'partial_fit':
            raise ValueError(FIT_FUNC_ERR.format(repr(model), n_batches))

    iter_offset = 0
    for idx in range(n_batches):
        sample = run_sample_pipeline(action_data, **sample_pipeline_kwargs)
        fitter = getattr(model, fit_func)
        fit_args, fit_kwargs = final_on_sample_step(fitter, model, sample,
                                                    iter_offset,
                                                    classes=None,
                                                    get_y_func=None,
                                                    get_y_kwargs=None,
                                                    get_weight_func=None,
                                                    get_weight_kwargs=None,
                                                )
        model = fitter(*fit_args, **fit_kwargs)
        iter_offset += getattr(model, 'n_iter', 1)
    if post_fit_func is not None:
        return pff(model, sample.df)
    return model


