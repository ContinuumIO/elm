## Transform section of config

Models using `transform` primarily or `fit_transform` may be configured in a `transform` section of the config, allowing the transform model, such as PCA, to be used as a `transform` step in the `sample_pipeline` (TODO LINK)

Here is an example showing the `transform` section of the config is similar in format to the `train` section of config:

```
transform: {
  pca: {
    model_init_class: "sklearn.decomposition:IncrementalPCA",
    model_init_kwargs: {"n_components": 2},
    ensemble: no_ensemble,
    model_scoring: Null,
  }
}
```
The `transform` config above configures a model that can be referred to as `pca`, an incrementally fit PCA with 2 components (`model_init_kwargs`).  No scoring is done on the PCA (`model_scoring` is Null / None).

To use a `transform` model, a config `pipeline` section might look like:

```
pipeline:
- data_source: NPP_DSRF1KD_L2GD
  sample_pipeline:
  - {select_canvas: band_1}
  - {flatten: C}
  - {sklearn_preprocessing: require_positive}
  - {drop_na_rows: true}
  - {method: fit_transform, transform: pca}
  steps:
  - {method: partial_fit, train: kmeans}
  - {predict: kmeans}

```

The `pipeline` spec above uses a `data_source` (`NPP_DSRF1KD_L2GD`) to read HDF4 files (defined elsewhere TODO LINK) and names a `sample_pipeline` with steps inclusive of selecting the canvas of `band_1`, flattening the sample, forcing negative numbers to positive, dropping NaN rows, and calling fit_transform on the `pca` model configured in the `transform` section of config.  Note that `select_canvas` here forces all bands in the data to be interpretted at the width, height and resolution of `band_1` as `band_1` is defined in the `band_specs` of the data source (TODO LINK).

The `pipeline` above also requires a `train` section to have created a model called `kmeans` (TODO Link to train section of config).



