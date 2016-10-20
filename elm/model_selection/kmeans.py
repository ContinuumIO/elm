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


def kmeans_aic(model, X, **kwargs):
    '''AIC (Akaike Information Criterion) for k-means for model selection

    Parameters:
        model:  An elm.pipeline.Pipeline with KMeans or MiniBatchKMeans as
                final step in Pipeline
        X:      The X data that were just given to "fit", or "partial_fit"
        kwargs: placeholder - ignored
    Returns:
        AIC float

    '''

    k, m = model._estimator.cluster_centers_.shape
    n = X.flat.values.shape[0]
    d = model._estimator.inertia_
    aic =  d + 2 * m * k
    delattr(model._estimator, 'labels_')
    return aic

name_idx = 0
def _next_name():
    global name_idx
    n = 'kmeans-init-'.format(name_idx)
    name_idx += 1
    return n

def kmeans_model_averaging(models, best_idxes=None, **kwargs):
    '''Run a KMeans on multiple KMeans models for cases where
    the representative sample is larger than one training batch.

    Parameters:
        models:  list of (tag, elm.pipeline.Pipeline instance) tuples
                 where the final step in each of the Pipeline instances
                 is either KMeans or MiniBatchKmeans from sklearn.cluster
        best_idxes: integer indices of the best -> worst models in models
        kwargs:   Keyword arguments passed via "model_selection_kwargs":
                  drop_n:  how many models to drop each generation
                  evolve_n: how many models to create from clustering
                            on clusters from models
                  init_n:   how many models to make new, initializing
                            from the best model
                  ngen, generation: kwargs added by model_selection logic
                            to control behavior on last generation
    Returns:
        list of (tag, Pipeline instance) tuples
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
        models = [models[idx] for idx in best_idxes]
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
    new_kwargs0 = best._estimator.get_params()
    new_kwargs0['init'] = best._estimator.cluster_centers_
    new_pipe_kwargs = best.get_params()
    if evolve_n:
        centroids = np.concatenate(tuple(m._estimator.cluster_centers_ for name, m in models))
        meta_model = MiniBatchKMeans(**new_kwargs0)
        new_kwargs0['batch_size'] = centroids.shape[0]
        meta_model.fit(centroids)
        for idx in range(evolve_n):
            new_estimator = best.unfitted_copy()
            new_kwargs = copy.deepcopy(new_kwargs0)
            new_params = {'init': meta_model.cluster_centers_,
                          'n_init': 1,
                          'n_clusters': meta_model.cluster_centers_.shape[0]}
            new_estimator._estimator.set_params(**new_params)
            new_model = (_next_name(), new_estimator)
            new_models.append(new_model)
    for new in range(init_n):
        new = best.unfitted_copy()
        new.set_params(**new_pipe_kwargs)
        new_models.append(('new-kmeans-{}'.format(idx), new))
    return tuple(new_models) + tuple(models)

