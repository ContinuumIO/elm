
from deap import creator, base, tools
from deap.tools.emo import selNSGA2
import numpy as np

from iamlp.settings import delayed

toolbox = base.Toolbox()


@delayed
def distance(c1, c2):
    resids = (c1 - c2)
    return np.sqrt(np.sum(resids ** 2))

@delayed
def get_distance_matrix(centroids):

    distance_matrix = np.empty((centroids.shape[0], centroids.shape[0]), dtype=np.float64)
    for i in range(centroids.shape[0]):
        for j in range(0, i):
            distance_matrix[i, j] = distance_matrix[j, i] = distance(centroids[i,:], centroids[j,:])
    distance_matrix[np.diag_indices_from(distance_matrix)] = 0.
    return distance_matrix

@delayed
def pareto_front(objectives, take=1, weights=None):
    if weights is None:
        weights = (-1,) * objectives.shape[1]
    creator.create("FitnessMin", base.Fitness, weights=weights)
    creator.create('Individual', tuple, fitness=creator.FitnessMin)
    #toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    #toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register('evaluate', lambda x:tuple(xi for xi in x))
    objectives = [creator.Individual(tuple(objectives[idx, :])) for idx in range(objectives.shape[0])]
    return np.array(selNSGA2(objectives, take))

@delayed
def kmeans_model_averaging(models, no_shuffle=1, require_pcent=2):
    inertia = [(m.inertia_, idx) for idx, m in enumerate(models)]
    delayed(inertia.sort)(key=lambda x:x[0])
    best_idxes = [i[1] for i in inertia[:no_shuffle]]
    centroids = np.concatenate([model.cluster_centers_ for model in models])
    class_counts = np.concatenate([model.class_counts_ for model in models])
    within_class_var = np.concatenate([model.within_class_var_ for model in models])
    distance_matrix = get_distance_matrix(centroids)
    models_improved = [models[idx] for idx in best_idxes]
    centroid_idxes = list(range(centroids.shape[0]))
    for idx in range(no_shuffle, len(models)):
        model = models[idx]
        new_centroids = []
        while len(new_centroids) < model.cluster_centers_.shape[0]:
            cen_choice = int(np.random.randint(0, model.cluster_centers_.shape[0], 1))
            cen = model.cluster_centers_[cen_choice]
            new_centroids.append(cen)
            position = idx * model.cluster_centers_.shape[0] + cen_choice
            distance_col = distance_matrix[position, :]
            objectives = np.column_stack((distance_col,
                                          within_class_var,
                                          class_counts))
            pfront = pareto_front(objectives,
                                  take=len(new_centroids) + 1,
                                  weights=(-1, -1, 1))
            for row_idx in range(pfront.shape[0]):
                row = pfront[row_idx, :]
                if row[0] == 0.:
                    continue
                position = np.where(np.all(row == objectives))[0]
                cen = centroids[position, :]
                if not any(np.all(c == cen) for c in new_centroids):
                    new_centroids.append(cen)
                    break
        model.cluster_centers_ = np.array(new_centroids)
        model.cluster_centers_ = model.cluster_centers_[:model.cluster_centers_.shape[0], :]
        delattr(model, 'inertia_')
        delattr(model, 'counts_')
        delattr(model, 'labels_')
        models_improved.append(model)
    return models_improved
