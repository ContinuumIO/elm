from collections import namedtuple, Sequence
import copy
from functools import partial
import inspect
import logging

import dask
import numbers

from elm.model_selection.util import get_args_kwargs_defaults
from elm.model_selection.evolve import ea_setup
from elm.config import import_callable, parse_env_vars
from elm.model_selection.base import base_selection
from elm.model_selection.scoring import score_one_model
from elm.model_selection.sorting import pareto_front
from elm.sample_util.sample_pipeline import create_sample_from_data_source


logger = logging.getLogger(__name__)

NO_ENSEMBLE = {'init_ensemble_size': 1,
               'ngen': 1,
               'partial_fit_batches': 1,
               'saved_ensemble_size': 1,}

_next_idx = 0

def _next_name():
    global _next_idx
    n = 'ensemble-{}'.format(_next_idx)
    _next_idx += 1
    return n

def _validate_ensemble_members(models):
    err_msg = ('Failed to instantiate models as sequence of tuples '
               '(name, model) where model has a fit or '
               'partial_fit method.  ')
    if not models or not isinstance(models, Sequence):
        raise ValueError(err_msg + "Got {}".format(repr(models)))
    example = 'First item in models list: {}'.format(models[0])
    err_msg += example
    if not any(isinstance(m, Sequence) for m in models):
        # list of models with no tags - make some up
        return [(_next_name(), m) for m in models]
    if not all(len(m) == 2 and isinstance(m, tuple) for m in models):
        raise ValueError(err_msg)
    return models


def _run_model_selection(models, model_selection, model_selection_kwargs,
                         ngen, generation,
                         scoring_kwargs):
    model_selection_kwargs['ngen'] = ngen
    model_selection_kwargs['generation'] = generation
    scoring_kwargs = scoring_kwargs or {}
    score_weights = (scoring_kwargs or {}).get('score_weights') or None
    sort_fitness = scoring_kwargs.get('sort_fitness', model_selection_kwargs.get('sort_fitness')) or None
    if not sort_fitness:
        sort_fitness = pareto_front
    else:
        sort_fitness = import_callable(sort_fitness)
    logger.debug('base_selection {}'.format(repr((models, model_selection, sort_fitness, score_weights, model_selection_kwargs))))
    models = base_selection(models,
                            model_selection=model_selection,
                            sort_fitness=sort_fitness,
                            score_weights=score_weights,
                            **model_selection_kwargs)
    models = _validate_ensemble_members(models)
    return models

