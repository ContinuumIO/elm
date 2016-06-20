import numpy as np

from iamlp.settings import delayed, SERIAL_EVAL
from iamlp.partial_fit import partial_fit
from iamlp.model_averaging.kmeans import kmeans_model_averaging

@delayed
def _kmeans_add_within_class_var(model, df):
    n_clusters = model.n_clusters
    var = (model.cluster_centers_[model.labels_] - df.values) ** 2
    model.within_class_var_ = np.zeros(n_clusters, dtype=np.float64)
    for idx in range(n_clusters):
        model.within_class_var_[idx] = delayed(np.sum)(var[model.labels_ == idx])
    bc = delayed(np.bincount)(model.labels_)
    model.class_counts_ = np.zeros(model.cluster_centers_.shape[0])
    model.class_counts_[:bc.size] = bc
    model.class_pcents_ = model.class_counts_ / np.sum(model.class_counts_) * 100.
    return model

@delayed(pure=True)
def partial_fit_once(models, new_models, no_shuffle, partial_fit_kwargs):
    if models is not None:
        models = models[:no_shuffle] + new_models[no_shuffle:]
    else:
        models = new_models
    return [partial_fit(model, **partial_fit_kwargs) for model in models]


@delayed
def kmeans_ensemble(init_models,
                    output_tag,
                    n_generations=2,
                    no_shuffle=1,
                    partial_fit_kwargs=None):
    models = None
    partial_fit_kwargs = partial_fit_kwargs or {}
    for generation in range(n_generations):
        new_models = init_models(models)
        models = partial_fit_once(models,
                                  new_models,
                                  no_shuffle,
                                  partial_fit_kwargs)
        if not SERIAL_EVAL:
            models = models.compute()
        if generation < n_generations - 1:
            models = kmeans_model_averaging(models)
    return models

