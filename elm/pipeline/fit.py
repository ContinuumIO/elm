import copy
import numpy as np

from elm.config.dask_settings import delayed, SERIAL_EVAL
from elm.pipeline.sample_pipeline import run_sample_pipeline
from elm.pipeline.sample_pipeline import final_on_sample_step
from elm.config import import_callable
from elm.model_selection import get_args_kwargs_defaults


FIT_FUNC_ERR = ('Expected model {} to have a '
                '"partial_fit" method with batches_per_gen = {}.\n'
                'Set batches_per_gen to 1 or use model with "partial_fit" method')


def fit(model,
        action_data,
        fit_func='partial_fit',
        get_y_func=None,
        get_y_kwargs=None,
        get_weight_func=None,
        get_weight_kwargs=None,
        post_fit_func=None,
        batches_per_gen=2,
        fit_kwargs=None):
    '''fit calls partial_fit or fit on a model after running the sample_pipeline

    Params:

        model:  instantiated model like MiniBatchKmeans()
        action_data: from elm.pipeline.sample_pipeline:all_sample_ops
                     (list of tuples of 3 items: (func, args, kwargs))
        fit_func: which attribute to use on model, typically "fit" or "partial_fit"
        get_y_func: function which returns a Y sample for an X sample dataframe
        get_y_kwargs: kwargs for get_y_func
        post_fit_func: a function run after all partial_fit batches of single fit batch
        batches_per_gen: number of partial_fit batches.  Must be 1 if fit_func is "fit"
        fit_kwargs: kwargs passed to partial_fit or fit method of model

        '''
    if post_fit_func is not None:
        pff = import_callable(post_fit_func, True, post_fit_func)
    get_y_kwargs = get_y_kwargs or {}
    if batches_per_gen > 1:
        if not fit_func == 'partial_fit':
            raise ValueError(FIT_FUNC_ERR.format(repr(model), batches_per_gen))

    iter_offset = 0
    for idx in range(batches_per_gen):
        sample = run_sample_pipeline(action_data)
        fitter = getattr(model, fit_func)
        fit_args, fit_kwargs = final_on_sample_step(fitter, model, sample,
                                                    iter_offset,
                                                    fit_kwargs,
                                                    classes=None,
                                                    flatten=True,
                                                    get_y_func=get_y_func,
                                                    get_y_kwargs=get_y_kwargs,
                                                    get_weight_func=get_weight_func,
                                                    get_weight_kwargs=get_weight_kwargs,
                                                )
        model = fitter(*fit_args, **fit_kwargs)
        iter_offset += getattr(model, 'n_iter', 1)
    if post_fit_func is not None:
        return pff(model, fit_args[0])# fit_args[0] is X as flattened
    return model


