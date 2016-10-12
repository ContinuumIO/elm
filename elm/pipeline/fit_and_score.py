import copy
from functools import partial
import logging
import numbers

import numpy as np
from sklearn.cross_validation import cross_val_score

from elm.sample_util.sample_pipeline import final_on_sample_step
from elm.config import import_callable
from elm.model_selection import get_args_kwargs_defaults
from elm.model_selection.scoring import (score_one_model,
                                      make_scorer)


__all__ = ['fit_and_score']
logger = logging.getLogger(__name__)

FIT_FUNC_ERR = ('Expected model {} to have a '
                '"partial_fit" method with partial_fit_batches = {}.\n'
                'Set partial_fit_batches to 1 or use model with "partial_fit" method')

def fit_and_score(model, X, y=None, sample_weight=None,
                  partial_fit_batches=None,
                  classes=None,
                  scoring=None,
                  scoring_kwargs=None,
                  method='partial_fit',
                  fit_kwargs=None,
                  return_sample=False):

    if partial_fit_batches > 1:
        if not hasattr(model, 'partial_fit') and method == 'partial_fit':
            raise ValueError(FIT_FUNC_ERR.format(repr(model), partial_fit_batches))

    iter_offset = 0
    for idx in range(partial_fit_batches):
        logger.info('Partial fit batch {} of {} in '
                    'current ensemble'.format(idx + 1, partial_fit_batches))
        fitter = getattr(model, method, getattr(model, 'fit'))
        fit_args, fit_kwargs = final_on_sample_step(fitter, model, X,
                                                    iter_offset,
                                                    fit_kwargs,
                                                    classes=classes, # TODO these need to be passed in some cases
                                                    y=y,
                                                    sample_weight=sample_weight)
        logger.debug('fit_args {} fit_kwargs {}'.format(fit_args, fit_kwargs))
        out = fitter(*fit_args, **fit_kwargs)
        if out is not None: # allow fitter func to modify in place
                            # or return a fitted model
            model = out
        iter_offset += getattr(model, 'n_iter', 1)
    if return_sample:
        return model, fit_args
    return model

