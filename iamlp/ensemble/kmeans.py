import numpy as np

from iamlp.settings import delayed
from iamlp.partial_fit import partial_fit
from iamlp.model_averaging.kmeans import kmeans_model_averaging


def _kmeans_add_within_class_var(model, df):
    var = (model.cluster_centers_[model.labels_] - df.values) ** 2
    model.within_class_var_ = np.zeros(model.cluster_centers_.shape[0], dtype=np.float64)
    for idx in range(np.max(model.labels_)):
        model.within_class_var_[idx] = np.sum(var[model.labels_ == idx])
    bc = np.bincount(model.labels_)
    model.class_counts_ = np.zeros(model.cluster_centers_.shape[0])
    model.class_counts_[:bc.size] = bc
    model.class_pcents_ = model.class_counts_ / np.sum(model.class_counts_) * 100.


@delayed
def kmeans_ensemble(init_models,
                    output_tag,
                    n_generations=2,
                    **selection_kwargs):
    models = None
    for generation in range(n_generations):
        models = init_models(models)
        if hasattr(models[0], 'compute'):
            models = [m.compute() for m in models]
        if generation < n_generations - 1:
            models = kmeans_model_averaging([m[0] for m in models])
        else:
            models = [m[0] for m in models]
            models.sort(key=lambda x:x.inertia_)

    return models
