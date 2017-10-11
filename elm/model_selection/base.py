from __future__ import absolute_import, division, print_function, unicode_literals

'''
----------------------------

``elm.model_selection.base``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''

from collections import namedtuple
import copy
from functools import partial
import logging
import inspect

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from elm.config import import_callable

from elm.model_selection.sorting import pareto_front

logger = logging.getLogger(__name__)


def base_selection(params_list,
                   fitnesses,
                   model_selection,
                   sort_fitness=pareto_front,
                   score_weights=None,
                   cv_results=None,
                   **kwargs):

    '''
    Select next parameters based on previous params_list, model_selection
    function and fitnesses.

    Parameters
    ----------
    params_list: list of dictionary of estimator parameter dicts
    fitnesses: 2D numpy array with columns as different objectives
    model_selection: callable.  If sort_fitness is given then the signature
        is expected:
            model_selection(params_list, best_idxes, **kwargs)

        otherwise the signature should be:

            model_selection(params_list, **kwargs)
    sort_fitness: defaults to pareto_front from deap
    score_weights: None for single objective optimization or a sequence for
        multiobjective, e.g. [1, -1, 1] for mminimize, maximize, minimize
    cv_results: cross validation results (from generations 0 - current).
        Typically the .cv_results_ attribute of EaSearchCV instance or similar
    **kwargs:  passed to model_selection

    '''
    logger.debug('base_selection with kwargs: {}'.format(kwargs))
    if sort_fitness == 'pareto_front':
        sort_fitness = pareto_front
    if not model_selection:
        return params_list
    kwargs = kwargs or {}
    if score_weights is None:
        score_weights = (1,)
    if sort_fitness is not None:
        if fitnesses.shape[1] == len(params_list) and fitnesses.shape[0] != len(params_list):
            fitnesses = fitnesses.T
        if fitnesses.shape[0] != len(params_list) or len(fitnesses.shape) != 2:
            raise ValueError('Expected scorer to return a scalar or 1-d array. '
                             'Found shape: {}'.format(fitnesses.shape))
        if fitnesses.shape[1] != len(score_weights):
            raise ValueError('Length of score_weights {} does '
                             'not match fitnesses.shape[1] {}'.format(fitnesses.shape[1], len(score_weights)))
        best_idxes = sort_fitness(score_weights, fitnesses)
        params_list = model_selection(params_list, best_idxes, **kwargs)
    else:
        params_list = model_selection(params_list, **kwargs)
    return params_list
