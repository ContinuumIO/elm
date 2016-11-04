'''Internal helpers for elm.pipeline'''
from collections import Sequence

from elm.model_selection.evolve import ea_setup
from elm.config import import_callable
from elm.model_selection.base import base_selection
from elm.model_selection.sorting import pareto_front


NO_ENSEMBLE = {'init_ensemble_size': 1,
               'ngen': 1,
               'partial_fit_batches': 1,
               'saved_ensemble_size': 1,}

_next_idx = 0

def _next_name(token):
    '''name in a dask graph'''
    global _next_idx
    n = '{}-{}'.format(token, _next_idx)
    _next_idx += 1
    return n

def _validate_ensemble_members(models):
    '''Take a list of estimators or a list of (tag, estimator) tuples
    Return (tag, estimator) tuples list'''
    err_msg = ('Failed to instantiate models as sequence of tuples '
               '(name, model) where model has a fit or '
               'partial_fit method.  ')
    if not models or not isinstance(models, Sequence):
        raise ValueError(err_msg + "Got {}".format(repr(models)))
    example = 'First item in models list: {}'.format(models[0])
    err_msg += example
    if not any(isinstance(m, Sequence) for m in models):
        # list of models with no tags - make some up
        return [(_next_name('ensemble_member'), m) for m in models]
    if not all(len(m) == 2 and isinstance(m, tuple) for m in models):
        raise ValueError(err_msg)
    return models


def _run_model_selection(models, model_selection, model_selection_kwargs,
                         ngen, generation,
                         scoring_kwargs):
    '''Run a model selection after adding ngen and generation to kwargs
    and finding the right sorting function for fitness

    Returns:
        list of (tag, model) tuples'''
    model_selection_kwargs['ngen'] = ngen
    model_selection_kwargs['generation'] = generation
    scoring_kwargs = scoring_kwargs or {}
    score_weights = (scoring_kwargs or {}).get('score_weights', model_selection_kwargs.get('score_weights'))
    sort_fitness = scoring_kwargs.get('sort_fitness', model_selection_kwargs.get('sort_fitness')) or None
    if not sort_fitness:
        sort_fitness = pareto_front
    else:
        sort_fitness = import_callable(sort_fitness)
    kw = {k: v for k,v in model_selection_kwargs.items()
          if not k in ('score_weights',)}
    models = base_selection(models,
                            model_selection=model_selection,
                            sort_fitness=sort_fitness,
                            score_weights=score_weights,
                            **kw)
    models = _validate_ensemble_members(models)
    return models

