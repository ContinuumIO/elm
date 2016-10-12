from functools import partial

import numpy as np
import sklearn.feature_selection as skfeat
import xarray as xr

from elm.config import import_callable
from elm.sample_util.sample_pipeline import check_array
from elm.readers import *
from elm.model_selection.util import get_args_kwargs_defaults



def feature_selection_base(X,
                          y=None,
                          **selection_dict):
    '''feature_selection_base returns indices of columns to keep
    Params:
        sample:  a data frame sample with column names used in the
                 keyword argument "keep_columns" (see below)
        selection_dict: dict with keys /values of:
            selection: a string like
                      "sklearn.feature_selection:VarianceThreshold"
            kwargs:   init key word arguments to selection given
            scoring: scoring, if needed, passed to selection().fit
            choices:    limit the feature selection to list of column names
                        (exclude metadata columns from thresholding)
        keep_columns: columns to keep regardless of selection's selection
    Returns:
        X: dataframe subset of selection's chosen columns
                   among choices
    '''
    keep_columns = selection_dict.get('keep_columns') or []
    feature_selection = selection_dict['selection']
    if feature_selection == 'all':
        return X
    feature_selection_kwargs = selection_dict.get('kwargs') or {}
    if not feature_selection_kwargs:
        feature_selection_kwargs = {k: v for k,v in selection_dict.items()
                                    if k != 'selection'}
    scoring_kwargs = selection_dict.get('scoring_kwargs') or {}
    feature_choices = selection_dict.get('choices') or 'all'
    if feature_choices == 'all':
        feature_choices = list(X.flat.band)
    feature_selection = import_callable(feature_selection)
    feature_scoring = selection_dict.get('scoring')
    if feature_scoring is not None:
        if isinstance(feature_scoring, str):
            feature_scoring = getattr(skfeat, feature_scoring, None)
        else:
            feature_scoring = import_callable(feature_scoring)
        if scoring_kwargs:
            feature_scoring = partial(feature_scoring, **scoring_kwargs)
        feature_selection_args = (feature_scoring,)
    else:
        feature_selection_args = ()
    selection = feature_selection(*feature_selection_args,
                                  **feature_selection_kwargs)
    band_idx = np.array([idx for idx, band in enumerate(X.flat.band)
                         if band in feature_choices])
    subset = X.flat[:, band_idx]
    check_array(subset.values, 'feature_selection:{} X subset'.format(selection))

    required_args, _, _ = get_args_kwargs_defaults(selection.fit)
    if ('y' in required_args or 'Y' in required_args):
        msg = ('feature_selection [{}] requires '
               'Y data but the sample_pipeline has not '
               'positioned a {{"get_y": True}} action before '
               'feature_selection'.format(feature_selection,))
        check_array(y, msg, ensure_2d=False)

    selection.fit(subset.values, y=y)
    ml_columns = selection.get_support(indices=True)
    X_dropped_bands =  ElmStore({'flat': xr.DataArray(X.flat[:, band_idx[ml_columns]].copy(),
                                              coords=[('space', X.flat.space),
                                                      ('band', X.flat.band[band_idx[ml_columns]])],
                                              dims=('space','band'),
                                              attrs=X.flat.attrs)},
                                          attrs=X.attrs)
    del X
    return X_dropped_bands
