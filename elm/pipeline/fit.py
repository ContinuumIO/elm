import copy
import numpy as np

from elm.samplers import random_images_selection
from elm.config.dask_settings import delayed, SERIAL_EVAL
from elm.pipeline.sample_util import run_sample_pipeline
from elm.pipeline.model_util import final_on_sample_step
from elm.config import import_callable


FIT_FUNC_ERR = ('Expected model {} to have a '
                '"partial_fit" method with n_batches = {}.\n'
                'Set n_batches to 1 or use model with "partial_fit" method')

def fit(model,
        action_data,
        fit_func='partial_fit',
        get_y_func=None,
        get_y_kwargs=None,
        post_fit_func=None,
        n_batches=2,
        fit_kwargs=None,
        **sample_pipeline_kwargs):
    '''fit calls partial_fit or fit on a model after running the sample_pipeline

    Params:

        model:  instantiated model like MiniBatchKmeans()
        action_data: from elm.pipeline.sample_util:all_sample_ops
                     (list of tuples of 3 items: (func, args, kwargs))
        fit_func: which attribute to use on model, typically "fit" or "partial_fit"
        get_y_func: function which returns a Y sample for an X sample dataframe
        get_y_kwargs: kwargs for get_y_func
        post_fit_func: a function run after all partial_fit batches of single fit batch
        n_batches: number of partial_fit batches.  Must be 1 if fit_func is "fit"
        fit_kwargs: kwargs passed to partial_fit or fit method of model
        **sample_pipeline_kwargs: params that need to be passed to run the sample
                  pipeline (TODO consider whether this is the best approach)

        '''
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
        sample = run_sample_pipeline(action_data)
        fitter = getattr(model, fit_func)
        fit_args, fit_kwargs = final_on_sample_step(fitter, model, sample,
                                                    iter_offset,
                                                    fit_kwargs,
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


