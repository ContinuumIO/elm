import array
from collections import namedtuple
import copy
import inspect
from itertools import product

from deap import creator, base, tools
from deap.tools.emo import selNSGA2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from elm.config import delayed
from elm.model_selection.util import (get_args_kwargs_defaults,
                                      filter_kwargs_to_func)
from elm.pipeline.sample_pipeline import flatten_data_arrays
from elm.sample_util.elm_store import (ElmStore,
                                       data_arrays_as_columns)



def kmeans_aic(model, x, y_true=None, scoring=None, **kwargs):
    '''AIC (Akaike Information Criterion) for k-means for model selection

    Parameters:
        model:  Typically KMeans or IncrementalKmeans instance from sklearn.cluster
        x:      The X data that were just given to "fit", or "partial_fit"
        y:      None (placeholder)
        scoring:None (placeholder)
        kwargs: ignored
    Returns:
        aic float

    '''
    assert x.ndim == 2
    k, m = model.cluster_centers_.shape
    n = x.shape[0]
    d = model.inertia_
    aic =  d + 2 * m * k
    return aic


def kmeans_model_averaging(models, best_idxes=None, **kwargs):

    linear_prob = np.linspace(len(models), 1, len(models))
    drop_n = kwargs['drop_n']
    evolve_n = kwargs['evolve_n']
    init_n = kwargs['init_n']
    last_gen = kwargs['ngen'] - 1 == kwargs['generation']
    if last_gen:
        # To avoid attempting to predict
        # when the model has not been fit
        # do not initialize any models on the
        # final generation of ensemble training.
        return models
    if drop_n > len(models):
        raise ValueError('All models would be dropped by drop_n {} '
                         'with len(models) {}'.format(drop_n, len(models)))
    if drop_n - evolve_n - init_n != 0:
        raise ValueError('Length of models should stay the same (drop_n - evolve_n - init_n === 0)')

    dropped = models[drop_n:]
    models = models[:-drop_n]
    centroids = np.concatenate(tuple(m.cluster_centers_ for name, m in models))
    name_idx = 0
    new_models = []
    for new in range(evolve_n):
        meta_model = KMeans(**filter_kwargs_to_func(KMeans, **kwargs['model_init_kwargs']))
        meta_model.fit(centroids)
        new_kwargs = copy.deepcopy(kwargs['model_init_kwargs'])
        new_kwargs['init'] = meta_model.cluster_centers_
        new_kwargs = filter_kwargs_to_func(KMeans, **new_kwargs)
        new_model = (dropped[name_idx][0], MiniBatchKMeans(**new_kwargs))
        new_models.append(new_model)
        name_idx += 1
    for new in range(init_n):
        new_kwargs = filter_kwargs_to_func(MiniBatchKMeans, **kwargs['model_init_kwargs'])
        new_models.append((dropped[name_idx][0],
                           MiniBatchKMeans(**new_kwargs)))
        name_idx += 1
    return tuple(new_models) + tuple(models)
