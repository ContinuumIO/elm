import numpy as np

from iamlp.settings import delayed
from iamlp.partial_fit import partial_fit
from iamlp.writers.serialize import serialize

def _kmeans_add_within_class_var(model, df):
    var = (model.cluster_centers_[model.labels_] - df.values) ** 2
    model.within_class_var = np.zeros(model.cluster_centers_.shape[0], dtype=np.float64)
    for idx in range(np.max(model.labels_)):
        model.within_class_var[idx] = np.sum(var[model.labels_ == idx])
    model.class_counts_ = np.bincount(model.labels_)
    model.class_pcents_ = model.class_counts_ / np.sum(model.class_counts_) * 100.

@delayed
def kmeans_ensemble(models,
                    output_tag,
                    filenames_gen,
                    band_specs,
                    n_ensemble=6,
                    n_samples_to_partial_fit=2,
                    n_per_file=100000,
                    files_per_sample=10,
                    **selection_kwargs):

    for ensemble_idx in range(n_ensemble):
        models_df = [partial_fit(model,
                          filenames_gen,
                          band_specs,
                          n_samples_to_partial_fit=n_samples_to_partial_fit,
                          n_per_file=n_per_file,
                          files_per_sample=files_per_sample,
                          **selection_kwargs) for model in models]
        dfs = [x[1] for x in models_df]
        models = [x[0] for x in models_df]
        del models_df
        if ensemble_idx < n_ensemble - 1:
            for df, model in zip(dfs, models):
                _kmeans_add_within_class_var(model, df)
            models = kmeans_model_averaging(models)
        else:
            models.sort(key=lambda x:x.inertia_)
    for model_idx, model in enumerate(models):
        serialize(output_tag + '_{}'.format(model_idx), model)
    return models
