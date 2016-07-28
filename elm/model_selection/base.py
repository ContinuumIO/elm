import array
from collections import namedtuple
import copy
from functools import partial
import logging
import inspect

from deap import creator, base, tools
from deap.tools.emo import selNSGA2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from elm.config import import_callable

from elm.model_selection.util import (get_args_kwargs_defaults,
                                     filter_kwargs_to_func)
from elm.model_selection.metrics import METRICS
import sklearn.metrics as sk_metrics

logger = logging.getLogger(__name__)

def pareto_front(weights, objectives, take=None):
    toolbox = base.Toolbox()
    take = take or objectives.shape[0]
    creator.create("FitnessMulti", base.Fitness, weights=weights)
    creator.create("Individual",
                   array.array,
                   typecode='d',
                   fitness=creator.FitnessMulti)
    toolbox.register('evaluate', lambda x: x)
    objectives = [creator.Individual(objectives[idx, :])
                  for idx in range(objectives.shape[0])]
    for (idx, obj) in enumerate(objectives):
        obj.idx = idx
        obj.fitness.values = toolbox.evaluate(obj)
    sel = selNSGA2(objectives, take)
    return tuple(item.idx for item in sel)


def no_selection(models, *args, **kwargs):
    return models


def make_scorer(scoring, **scoring_kwargs):
    if not hasattr(scoring, 'fit'):
        if scoring in METRICS:
            scoring = import_callable(METRICS[scoring])
        else:
            scoring = import_callable(scoring)
    func_kwargs = filter_kwargs_to_func(scoring, **scoring_kwargs)
    scorer = sk_metrics.make_scorer(scoring,
                                 greater_is_better=scoring_kwargs.get('greater_is_better', True),
                                 needs_proba=scoring_kwargs.get('needs_proba', False),
                                 needs_threshold=scoring_kwargs.get('needs_threshold', False),
                                 **func_kwargs)
    return scorer


def _score_one_model_with_y_true(model,
                                scoring,
                                x,
                                y_true,
                                sample_weight=None,
                                **kwargs):
    if not isinstance(scoring, sk_metrics.scorer._PredictScorer):
        scorer = make_scorer(scoring, **kwargs)
    else:
        scorer = scoring
    # now scorer has signature:
    #__call__(self, estimator, x, y_true, sample_weight=None)
    return scorer(model, x, y_true, sample_weight=sample_weight)



def _score_one_model_no_y_true(model,
                               scoring,
                               x,
                               sample_weight=None,
                               **kwargs):
    kwargs_to_scoring = copy.deepcopy(kwargs)
    kwargs_to_scoring['sample_weight'] = sample_weight
    kwargs_to_scoring = filter_kwargs_to_func(scoring,
                                            **kwargs_to_scoring)
    return scoring(model, x, **kwargs_to_scoring)


def select_top_n_models(models, best_idxes, **kwargs):
    '''Run in ensemble without modifying the models on each generation, then
    on final generation take "top_n" models (top_n from kwargs)'''
    top_n = kwargs['top_n']
    logger.debug('Enter select_top_n with {} models and best_idxes {}'.format(len(models), best_idxes))
    if kwargs['generation'] < kwargs['n_generations'] - 1:
        pass
    else:
        models = [models[b] for b in best_idxes[:top_n]]
    logger.debug('Exit select_top_n with {} models'.format(len(models)))
    return models

def score_one_model(model,
                    scoring,
                    x,
                    y_true=None,
                    sample_weight=None,
                    **kwargs):
    if y_true is not None:
        return _score_one_model_with_y_true(model,
                    scoring,
                    x,
                    y_true=y_true,
                    sample_weight=None,
                    **kwargs)
    scoring = import_callable(scoring)
    return _score_one_model_no_y_true(model,
                    scoring,
                    x,
                    sample_weight=None,
                    **kwargs)


def base_selection(models,
                   model_selection_func=None,
                   sort_fitness=pareto_front,
                   score_weights=None,
                   **model_selection_kwargs):
    logger.debug('base_selection with kwargs: {}'.format(model_selection_kwargs))
    if sort_fitness == 'pareto_front':
        sort_fitness = pareto_front
    if not model_selection_func or model_selection_func == 'no_selection':
        return models
    model_selection_kwargs = model_selection_kwargs or {}
    if score_weights is not None:
        if score_weights is None or not hasattr(score_weights, '__len__'):
            raise ValueError('Expected score_weights keyword arg (array of '
                            '-1 for minmize, 1 for maximize for each scoring matrix column)')
        # scoring has signature: scoring(y, y_pred, **kwargs).
        scores = [model._score for name, model in models]
        scores = np.atleast_2d(np.array(scores))
        if scores.shape[1] == len(models) and scores.shape[0] != len(models):
            scores = scores.T
        if scores.shape[0] != len(models) or len(scores.shape) != 2:
            raise ValueError('Expected scorer to return a scalar or 1-d array. '
                             'Found shape: {}'.format(scores.shape))
        if scores.shape[1] != len(score_weights):
            raise ValueError('Length of score_weights {} does '
                             'not match scores.shape[1] {}'.format(scores.shape[1], len(score_weights)))
        best_idxes = sort_fitness(score_weights, scores)

        models = model_selection_func(models, best_idxes, **model_selection_kwargs)
    else:
        models = model_selection_func(models, **model_selection_kwargs)
    return models

