from functools import partial

import numpy as np
import sklearn.feature_selection as skfeat
import xarray as xr

from elm.config import import_callable
from elm.pipeline.sample_pipeline import (flatten_cube,
                                          flattened_to_cube,)

def feature_selection_base(sample_x,
                          selection_dict,
                          sample_y=None,
                          keep_columns=None):
    '''feature_selection_base returns indices of columns to keep
    Params:
        sample:  a data frame sample with column names used in the
                 keyword argument "keep_columns" (see below)
        selection_dict: dict with keys /values of:
            selection: a string like
                      "sklearn.feature_selection:VarianceThreshold"
            kwargs:   init key word arguments to selection given
            score_func: score_func, if needed, passed to selection().fit
            choices:    limit the feature selection to list of column names
                        (exclude metadata columns from thresholding)
        keep_columns: columns to keep regardless of selection's selection
    Returns:
        sample_x: dataframe subset of selection's chosen columns
                   among choices
    '''
    if len(sample_x.sample.shape) == 3:
        sample_x = flatten_cube(sample_x)
        reshape_needed = True
    else:
        reshape_needed = False
    feature_selection = selection_dict['selection']
    if feature_selection == 'all':
        return sample_x
    feature_selection_kwargs = selection_dict['kwargs']
    score_func_kwargs = selection_dict.get('score_func_kwargs') or {}
    feature_choices = selection_dict['choices']
    feature_selection = import_callable(feature_selection)
    feature_score_func = selection_dict.get('score_func')
    if feature_score_func is not None:
        if isinstance(feature_score_func, str):
            feature_score_func = getattr(skfeat, feature_score_func, None)
        else:
            feature_score_func = import_callable(selection_dict['score_func'])
        if score_func_kwargs:
            feature_score_func = partial(feature_score_func, **score_func_kwargs)
        feature_selection_args = (feature_score_func,)
    else:
        feature_selection_args = ()
    selection = feature_selection(*feature_selection_args,
                                **feature_selection_kwargs)
    if feature_choices == 'all':
        feature_choices = list(sample_x.sample.band)
    band_idx = np.array([idx for idx, band in enumerate(sample_x.sample.band)
                         if band in feature_choices])
    subset = sample_x.sample[:, band_idx]
    selection.fit(subset.values, y=sample_y)
    ml_columns = selection.get_support(indices=True)
    sample_x_dropped_bands =  xr.Dataset({'sample': xr.DataArray(sample_x.sample[:, band_idx[ml_columns]].copy(),
                                              coords=[('space', sample_x.sample.space),
                                                      ('band', band_idx[ml_columns])],
                                              dims=('space','band'),
                                              attrs=sample_x.sample.attrs)},
                                          attrs=sample_x.attrs)
    del sample_x
    if reshape_needed:
        return flattened_to_cube(sample_x_dropped_bands)
    return sample_x_dropped_bands
