import copy
from functools import partial
import logging
import numbers

import numpy as np
from sklearn.cross_validation import cross_val_score

from elm.config.dask_settings import delayed, SERIAL_EVAL
from elm.pipeline.sample_pipeline import run_sample_pipeline
from elm.pipeline.sample_pipeline import final_on_sample_step
from elm.config import import_callable
from elm.model_selection import get_args_kwargs_defaults
from elm.model_selection.scoring import (score_one_model,
                                      make_scorer)

logger = logging.getLogger(__name__)

FIT_FUNC_ERR = ('Expected model {} to have a '
                '"partial_fit" method with batches_per_gen = {}.\n'
                'Set batches_per_gen to 1 or use model with "partial_fit" method')


def fit(model,
        action_data,
        get_y_func=None,
        get_y_kwargs=None,
        get_weight_func=None,
        get_weight_kwargs=None,
        batches_per_gen=2,
        fit_kwargs=None,
        scoring=None,
        scoring_kwargs=None,
        transform_dict=None,
        fit_method='partial_fit'):
    '''fit calls partial_fit or fit on a model after running the sample_pipeline

    Params:

        model:  instantiated model like MiniBatchKmeans()
        action_data: from elm.pipeline.sample_pipeline:get_sample_pipeline_action_data
                     (list of tuples of 3 items: (func, args, kwargs))
        get_y_func: function which returns a Y sample for an X sample dataframe
        get_y_kwargs: kwargs for get_y_func
        batches_per_gen: number of partial_fit batches.  Must be 1 if fit_func is "fit"
        fit_kwargs: kwargs passed to partial_fit or fit method of model

    '''
    get_y_kwargs = get_y_kwargs or {}
    if batches_per_gen > 1:
        if not hasattr(model, 'partial_fit') and fit_method == 'partial_fit':
            raise ValueError(FIT_FUNC_ERR.format(repr(model), batches_per_gen))

    iter_offset = 0
    for idx in range(batches_per_gen):
        logger.info('Partial fit batch {} of {} in '
                    'current ensemble'.format(idx + 1, batches_per_gen))
        sample, sample_y, sample_weight = run_sample_pipeline(action_data, transform_dict=transform_dict)
        fitter = getattr(model, fit_method)
        fit_args, fit_kwargs = final_on_sample_step(fitter, model, sample,
                                                    iter_offset,
                                                    fit_kwargs,
                                                    classes=None,
                                                    flatten=True,
                                                    sample_y=sample_y,
                                                    sample_weight=sample_weight,
                                                )
        model = fitter(*fit_args, **fit_kwargs)
        if scoring:
            fit_kwargs.update(scoring_kwargs or {})
            kw = {k:v for k,v in fit_kwargs.items() if not k in ('scoring',)}
            model = score_one_model(model, scoring, *fit_args, **kw)
        iter_offset += getattr(model, 'n_iter', 1)
    return model


