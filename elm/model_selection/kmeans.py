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
from elm.model_selection.util import (get_args_kwargs_defaults,
                                      filter_kwargs_to_func)


def kmeans_aic(model, X, y_true=None, scoring=None, **kwargs):
    '''AIC (Akaike Information Criterion) for k-means for model selection

    Parameters:
        model:  An elm.pipeline.Pipeline with a KMeans _estimator
        x:      The X data that were just given to "fit", or "partial_fit"
        y:      None (placeholder)
        scoring:None (placeholder)
        kwargs: ignored
    Returns:
        aic float

    '''

    k, m = model._estimator.cluster_centers_.shape
    n = X.flat.values.shape[0]
    d = model._estimator.inertia_
    aic =  d + 2 * m * k
    delattr(model._estimator, 'labels_')
    return aic


def kmeans_model_averaging(models, best_idxes=None, **kwargs):
    '''
    models:  list of elm.pipeline.Pipeline with a KMeans _estimator
    '''

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


    if drop_n:
        dropped = models[-drop_n:]
        models = models[:-drop_n]

    name_idx = 0
    new_models = []
    best = models[0][1]
    good_n_clusters = [model._estimator.get_params()['n_clusters']
                       for tag, model in models]
    new_kwargs0 = best._estimator.get_params()
    new_pipe_kwargs = best.get_params()
    if evolve_n:
        centroids = np.concatenate(tuple(m._estimator.cluster_centers_ for name, m in models))
        names = [name for name, model in models]
        meta_model = MiniBatchKMeans(**new_kwargs0)
        new_kwargs0['batch_size'] = centroids.shape[0]
        meta_model.fit(centroids)
        for name_idx, new in enumerate(range(evolve_n)):
            new_estimator = copy.deepcopy(best)
            new_kwargs = copy.deepcopy(new_kwargs0)
            new_params = {'init': meta_model.cluster_centers_, 'n_init': 1}
            new_estimator._estimator.set_params(**new_params)
            new_model = (names[name_idx], new_estimator)
            new_models.append(new_model)
    for idx, new in enumerate(range(init_n)):
        new = copy.deepcopy(models[0][1])
        new.set_params(**new_pipe_kwargs)
        new_models.append(('new-kmeans-{}'.format(idx), new))
    return tuple(new_models)

