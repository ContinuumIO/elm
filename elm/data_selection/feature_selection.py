from functools import partial

import sklearn.feature_selection as skfeat
from elm.config import import_callable


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
    if sample_y is not None:
        y = sample_y.values[:, 0]  # TODO is y always 1 column?
    else:
        y = None
    feature_selection = selection_dict['selection']
    feature_selection_kwargs = selection_dict['kwargs']
    feature_score_func = selection_dict['score_func']
    score_func_kwargs = selection_dict.get('score_func_kwargs') or {}
    feature_choices = selection_dict['choices']
    if feature_selection == 'all':
        return sample_x
    feature_selection = import_callable(feature_selection)
    feature_selection_kwargs = feature_selection_kwargs or {}
    if feature_score_func is not None:
        if not callable(feature_score_func):
            func = getattr(skfeat, feature_score_func, None)
            if func is None:
                func = import_callable(feature_score_func)
            feature_score_func = func
        if score_func_kwargs:
            feature_score_func = partial(feature_score_func, **score_func_kwargs)
        feature_selection_args = (feature_score_func,)
    else:
        feature_selection_args = ()
    selection = feature_selection(*feature_selection_args,
                                **feature_selection_kwargs)
    if feature_choices == 'all':
        feature_choices = sample_x.columns
    x = sample_x[feature_choices].values
    selection.fit(x, y=y)
    ml_columns = [sample_x[feature_choices].columns[idx]
                  for idx in selection.get_support(indices=True)]
    keep_columns = keep_columns or []
    keep_columns = list(keep_columns)
    select_column_idxes = list(ml_columns + keep_columns)
    return sample_x[select_column_idxes]
