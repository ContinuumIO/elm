from __future__ import absolute_import, division, print_function

'''
----------------------------

``elm.model_selection.kmeans``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

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
import xarray as xr

from xarray_filters.func_signatures import (get_args_kwargs_defaults,
                                            filter_args_kwargs)


def kmeans_aic(model, X, **kwargs):
    '''AIC (Akaike Information Criterion) for k-means for model selection

    Parameters:
        :model:  An elm.pipeline.Pipeline with KMeans or MiniBatchKMeans as final step in Pipeline
        :X:      The X data that were just given to "fit", or "partial_fit"
        :kwargs: placeholder - ignored

    Returns:
        :AIC: float

    '''

    k, m = model._estimator.cluster_centers_.shape
    if isinstance(X, xr.DataArray):
        n = X.flat.values.shape[0]
    else:
        n = X.shape[0]
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
        :models:  list of (tag, elm.pipeline.Pipeline instance) tuples /
                 where the final step in each of the Pipeline instances /
                 is either KMeans or MiniBatchKmeans from sklearn.cluster
        :best_idxes: integer indices of the best -> worst models in models
        :kwargs:   Keyword arguments passed via "model_selection_kwargs":

                  * :drop_n:  how many models to drop each generation
                  * :evolve_n: how many models to create from clustering on clusters from models
                  * :ngen, generation: kwargs added by model_selection logic to control behavior on last generation
                  * :reps: repititions in meta-clustering (see below)

    Returns:
        :list of tuples: of (tag, Pipeline instance) tuples

    Notes:

        * First the drop_n worst AIC scoring (tag, Pipeline) tuples are dropped
        * The for repeat in range evolve_n:

            * Bootstrap existing centroids with linear probability density /
              related to AIC score rank, preferring lower AICscores
            * Run a K-Means on those centroids
            * Initialize a new pipeline with those params

        * The number of (tag, Pipeline) tuples is `len(models) + evolve_n - drop_n`

    '''

    drop_n = kwargs.get('drop_n', 0)
    evolve_n = kwargs.get('evolve_n', 0)
    reps = kwargs.get('reps', 100)
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

    def get_shape():
        '''Get a random centroids shape from linear prob density'''
        probs = np.linspace(len(models) + 1, 1, len(models))
        probs /= probs.sum()
        idx = np.random.choice(np.arange(len(models)), p=probs)
        _, m = models[idx]
        return m._estimator.cluster_centers_.shape

    def get_best(shp):
        '''Get the best of a given shape of centroids

        The logic in these two functions is intended to avoid
        selecting a model that has a different input
        feature column dimension shape, such as Pipelines
        with PCA before K-Means
        '''
        choices = [(tag, model) for tag, model in models
                   if model._estimator.cluster_centers_.shape == shp]
        assert choices, repr(models)
        probs = np.linspace(len(choices) + 1, 1, len(choices))
        probs /= probs.sum()

        best_idx = np.random.choice(np.arange(len(choices)), p=probs)
        best_tag, best = choices[best_idx]
        if best._estimator.cluster_centers_.shape != shp:
            return get_best(shp, simple=simple)
        return best

    for idx in range(evolve_n):
        shp = get_shape()
        resampling = [get_best(shp) for _ in range(reps)]
        centroids = np.concatenate(tuple(m._estimator.cluster_centers_ for m in resampling))
        meta_model = resampling[0]
        new_params = meta_model._estimator.get_params()
        est = MiniBatchKMeans(**new_params)
        est.partial_fit(centroids)
        new_params.update({'init': est.cluster_centers_,
                          'n_init': 1})
        meta_model.steps[-1] = (meta_model.steps[-1][0], est)
        meta_model._estimator = est
        new_model = (_next_name(), meta_model)
        new_models.append(new_model)
    return tuple(new_models) + tuple(models)
