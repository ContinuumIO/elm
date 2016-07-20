import array
from collections import namedtuple
import copy
import inspect

from deap import creator, base, tools
from deap.tools.emo import selNSGA2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from elm.config import delayed
from elm.model_selection.util import get_args_kwargs_defaults
toolbox = base.Toolbox()

KmeansVar = namedtuple('KmeansVar',
                       ['model',
                        'sample',
                        'within_class_var',
                        'class_counts'])

def kmeans_add_within_class_var(model, flattened):
    n_clusters = model.cluster_centers_.shape[0]
    var = np.sum((model.cluster_centers_[model.labels_] - flattened) ** 2, axis=1)
    within_class_var = np.zeros(n_clusters, dtype=np.float64)
    df2 = pd.DataFrame({'var': var, 'label': model.labels_})
    g = df2.groupby('label')
    agg = g.sum()
    for idx in range(n_clusters):
        if idx in agg.index:
            sel = agg.loc[idx]
            within_class_var[idx] = sel.values[0]
        else:
            within_class_var[idx] = 0.
    bc = np.bincount(model.labels_)
    class_counts = np.zeros(model.cluster_centers_.shape[0])
    class_counts[:bc.size] =  bc
    return KmeansVar(model, flattened, within_class_var, class_counts)


def distance(c1, c2):
    resids = (c1 - c2)
    return np.sqrt(np.sum(resids ** 2))


def get_distance_matrix(centroids):

    distance_matrix = np.empty((centroids.shape[0], centroids.shape[0]), dtype=np.float64)
    for i in range(centroids.shape[0]):
        for j in range(0, i):
            distance_matrix[i, j] = distance_matrix[j, i] = distance(centroids[i,:], centroids[j,:])
    distance_matrix[np.diag_indices_from(distance_matrix)] = 0.
    return distance_matrix


def pareto_front(objectives, centroids, take, weights):
    creator.create("FitnessMulti", base.Fitness, weights=weights)
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMulti)
    toolbox.register('evaluate', lambda x: x)
    objectives = [creator.Individual(objectives[idx, :]) for idx in range(objectives.shape[0])]
    objectives2 = []
    for (idx, (obj, cen)) in enumerate(zip(objectives, centroids)):
        obj.idx = idx
        obj.cen = cen
        obj.fitness.values = toolbox.evaluate(obj)
    sel = selNSGA2(objectives, take)
    return tuple((item.cen, item.idx) for item in sel)

def _rand_int_exclude(exclude, start, end, step):
    choices = np.arange(start, end, step)
    probs = np.ones(choices.size)
    for e in exclude:
        probs[e] = 0
    probs /= probs.sum()
    return int(np.random.choice(choices, p=probs))

def kmeans_model_averaging(models, **kwargs):
    no_shuffle = kwargs['no_shuffle']
    model_init_class = kwargs['model_init_class']
    model_init_kwargs = kwargs['model_init_kwargs']
    n_clusters = model_init_kwargs['n_clusters']
    inertia = [(m.model.inertia_, idx) for idx, m in enumerate(models)]
    inertia.sort()
    best_idxes = [i[1] for i in inertia[:no_shuffle]]
    centroids = np.concatenate([m.model.cluster_centers_ for m in models])
    class_counts = np.concatenate([m.class_counts for m in models])
    within_class_var = np.concatenate([m.within_class_var for m in models])
    distance_matrix = get_distance_matrix(centroids)
    new_models = [models[idx].model for idx in best_idxes]
    centroid_idxes = list(range(centroids.shape[0]))
    num_rows = n_clusters * len(models)
    for idx in range(no_shuffle, len(models)):
        model_stats = models[idx]
        new_centroids = []
        exclude = set()
        for idx in range(n_clusters // 2 + 1):
            cen_choice = _rand_int_exclude(exclude, 0, num_rows, 1)
            exclude.add(cen_choice)
            cen = model_stats.model.cluster_centers_[cen_choice % n_clusters]
            new_centroids.append(cen)
            distance_col = distance_matrix[cen_choice, :]
            objectives = np.column_stack((distance_col,
                                          within_class_var,
                                          class_counts))
            best = pareto_front(objectives,
                                centroids,
                                  take=1,
                                  weights=(1, -1, 1))[0]
            row = best[0]
            position = best[1]
            exclude.add(position)
            new_centroids.append(np.array(row))
        cluster_centers_ = np.row_stack(new_centroids)
        cluster_centers_ = cluster_centers_[:n_clusters,:]
        kw = copy.deepcopy(model_init_kwargs)
        kw.update({'n_clusters':n_clusters, 'init': cluster_centers_})
        model = model_init_class(**kw)
        new_models.append(model)
    return new_models
