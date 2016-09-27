
## Sample pipeline (transformations on images loaded from files)

In the machine learning pipeline, the `data_source` is used to define a sampler function which in turn can return a sample of data when called.  This section of the documentation describes the `sample_pipeline` which in the config system is a list of actions to complete on each sample defined by the `data_source`.

Here is an example `sample_pipeline`:
```
pipeline:
- data_source: NPP_DSRF1KD_L2GD
  sample_pipeline:
  - {select_canvas: band_1}
  - {flatten: C}
  - {sklearn_preprocessing: require_positive}
  - {drop_na_rows: true}
  - {random_sample: 500}
  - {get_y: true}
  steps:
  - {method: fit, train: train1}
  - {predict: train1}
```
The `sample_pipeline` above draws samples from the `NPP_DSRF1KD_L2GD` data source then
 * samples them to the extent and resolution of `band_1` (the `select_canvas` step),
 * flattens the sample (`flatten`),
 * forces negative numbers to positive (`sklearn_preprocessing`),
 * drops NaN rows (`drop_na_rows`)
 * randomly samples for 500 rows
 * calls the `get_y` function from the `data_source` `NPP_DSRF1KD_L2GD` data source to get Y data of the same row shape as the X data after random sampling,
 * uses the final transformed sample for training and prediction

The example above would require a `train` config defining `train1` and an `sklearn_preprocessing` section of config defining `require_positive`.

Another action in the `sample_pipeline` may be `modify_coords`, e.g.:

```
{modify_coords: "mypackage.mymodule:xarray_edits"}
```
which names a function to be called on the current sample, returning a sample when called.