from functools import partial

import sklearn.feature_selection as skfeat
from iamlp.config import import_callable


def feature_selector_base(sample_x,
                          selector_dict,
                          sample_y=None,
                          keep_columns=None):
    '''feature_selector_base returns indices of columns to keep
    Params:
        sample:  a data frame sample with column names used in the
                 keyword argument "keep_columns" (see below)
        selector_dict: dict with keys /values of:
            selector: a string like
                      "sklearn.feature_selection:VarianceThreshold"
            kwargs:   init key word arguments to selector given
            score_func: score_func, if needed, passed to selector().fit
            choices:    limit the feature selection to list of column names
                        (exclude metadata columns from thresholding)
        keep_columns: columns to keep regardless of selector's selection
    Returns:
        sample_x: dataframe subset of selector's chosen columns
                   among choices
    '''
    if sample_y is not None:
        y = sample_y.values[:, 0]  # TODO is y always 1 column?
    else:
        y = None
    feature_selector = selector_dict['selector']
    feature_selector_kwargs = selector_dict['kwargs']
    feature_score_func = selector_dict['score_func']
    score_func_kwargs = selector_dict.get('score_func_kwargs') or {}
    feature_choices = selector_dict['choices']
    if feature_selector == 'all':
        return sample_x
    feature_selector = import_callable(feature_selector)
    feature_selector_kwargs = feature_selector_kwargs or {}
    if feature_score_func is not None:
        if feature_score_func in dir(skfeat):
            feature_score_func = getattr(skfeat, feature_score_func)
        else:
            feature_score_func = import_callable(feature_score_func)

        if score_func_kwargs:
            feature_score_func = partial(feature_score_func, **score_func_kwargs)
        feature_selector_args = (feature_score_func,)
    else:
        feature_selector_args = ()
    selector = feature_selector(*feature_selector_args,
                                **feature_selector_kwargs)
    if feature_choices == 'all':
        feature_choices = sample_x.columns
    x = sample_x[feature_choices].values
    selector.fit(x, y=y)
    ml_columns = [sample_x[feature_choices].columns[idx]
                  for idx in selector.get_support(indices=True)]
    keep_columns = keep_columns or []
    keep_columns = list(keep_columns)
    select_column_idxes = list(ml_columns + keep_columns)
    return sample_x[select_column_idxes]
