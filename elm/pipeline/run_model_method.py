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
                '"partial_fit" method with partial_fit_batches = {}.\n'
                'Set partial_fit_batches to 1 or use model with "partial_fit" method')


def run_model_method(model,
                     x_y_sample_weight,
                     partial_fit_batches=None,
                     classes=None,
                     scoring=None,
                     scoring_kwargs=None,
                     method='partial_fit',
                     fit_kwargs=None,
                     return_sample=False):
    '''fit - call a method of scikit-learn which may be fit, fit_transform, or transform
    Parameters:
        model: sklearn model instance
        x_y_sample_weight: tuple of (X, Y, sample_weight)
                                       X: ElmStore with DataArray "flat"
                                       Y: None or 1-d np.array
                                       sample_weight: None or 1-d np.array

        partial_fit_batches:  how many partial_fit_batches
        classes:              1-d integer np.array of all possible classes
        scoring:              scoring function, defaults to model.score()
        scoring_kwargs:       kwargs passed to "scoring" function
        method:           model's method to call, e.g. "fit" or "fit_transform"
        fit_kwargs:           kwargs passed to model's "method"
    '''

    if partial_fit_batches > 1:
        if not hasattr(model, 'partial_fit') and method == 'partial_fit':
            raise ValueError(FIT_FUNC_ERR.format(repr(model), partial_fit_batches))

    iter_offset = 0
    for idx in range(partial_fit_batches):
        logger.info('Partial fit batch {} of {} in '
                    'current ensemble'.format(idx + 1, partial_fit_batches))

        assert len(x_y_sample_weight) == 3, repr(len(x_y_sample_weight))
        sample, sample_y, sample_weight = x_y_sample_weight
        fitter = getattr(model, method, getattr(model, 'fit'))
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
    if return_sample:
        return model, fit_args
    return model

