import copy
from functools import partial
import logging
import numbers

import numpy as np
from sklearn.cross_validation import cross_val_score

from elm.sample_util.sample_pipeline import run_sample_pipeline
from elm.sample_util.sample_pipeline import final_on_sample_step
from elm.config import import_callable
from elm.model_selection import get_args_kwargs_defaults
from elm.model_selection.scoring import (score_one_model,
                                      make_scorer)

logger = logging.getLogger(__name__)

FIT_FUNC_ERR = ('Expected model {} to have a '
                '"partial_fit" method with batches_per_gen = {}.\n'
                'Set batches_per_gen to 1 or use model with "partial_fit" method')


def fit(model,
        sample_sample_y_sample_weight,
        batches_per_gen=None,
        fit_kwargs=None,
        classes=None,
        scoring=None,
        scoring_kwargs=None,
        transform_model=None,
        fit_method='partial_fit'):
    '''fit calls partial_fit or fit on a model after running the sample_pipeline

    Params:

        model:  instantiated model like MiniBatchKmeans()
        action_data: from elm.sample_util.sample_pipeline:get_sample_pipeline_action_data
                     (list of tuples of 3 items: (func, args, kwargs))
        get_y_func: function which returns a Y sample for an X sample dataframe
        get_y_kwargs: kwargs for get_y_func
        batches_per_gen: number of partial_fit batches.  Must be 1 if fit_func is "fit"
        fit_kwargs: kwargs passed to partial_fit or fit method of model

    '''
    logger.info((' - '.join(('{}',) * 9).format(model,
        sample_sample_y_sample_weight,
        batches_per_gen,
        fit_kwargs,
        classes,
        scoring,
        scoring_kwargs,
        transform_model,
        fit_method)))  #TODO remove this log
    if batches_per_gen > 1:
        if not hasattr(model, 'partial_fit') and fit_method == 'partial_fit':
            raise ValueError(FIT_FUNC_ERR.format(repr(model), batches_per_gen))

    iter_offset = 0
    for idx in range(batches_per_gen):
        logger.info('Partial fit batch {} of {} in '
                    'current ensemble'.format(idx + 1, batches_per_gen))

        if sample_sample_y_sample_weight is None:
            sample, sample_y, sample_weight = run_sample_pipeline(action_data, transform_model=transform_model)
        else:
            assert len(sample_sample_y_sample_weight) == 3, repr(len(sample_sample_y_sample_weight))
            sample, sample_y, sample_weight = sample_sample_y_sample_weight
        fitter = getattr(model, fit_method, getattr(model, 'fit'))
        fit_args, fit_kwargs = final_on_sample_step(fitter, model, sample,
                                                    iter_offset,
                                                    fit_kwargs,
                                                    classes=classes, # TODO these need to be passed in some cases
                                                    sample_y=sample_y,
                                                    sample_weight=sample_weight)
        logger.debug('fit_args {} fit_kwargs {}'.format(fit_args, fit_kwargs))
        out = fitter(*fit_args, **fit_kwargs)
        if out is not None: # allow fitter func to modify in place
                            # or return a fitted model
            model = out
        if scoring or scoring_kwargs:
            kw = copy.deepcopy(scoring_kwargs or {})
            kw.update(fit_kwargs)
            kw = {k: v for k,v in kw.items()
                  if not k in ('scoring',)}
            model = score_one_model(model, scoring, *fit_args, **kw)
        iter_offset += getattr(model, 'n_iter', 1)
    return model

