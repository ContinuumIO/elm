from functools import partial
import numpy as np
from sklearn.base import BaseEstimator
from dask.utils import derived_from
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.linear_model import LinearRegression as skLinearRegression
from xarray_filters.mldataset import MLDataset
from xarray_filters.pipeline import Step
import xarray as xr


from dask_searchcv.model_selection import DaskBaseSearchCV, _DOC_TEMPLATE
from elm.model_selection.ensemble import EnsembleCV
from elm.model_selection.sklearn_mldataset import _call_sk_method
from elm.model_selection.sklearn_mldataset import SklearnMixin

_hi_oneliner = """TODO
"""
_hi_description = """TODO
"""
_hi_parameters = """TODO
"""
_hi_example = """TODO
"""


def concat_features(method, preproc=None, postproc=None):
    #@if_delegate_has_method(delegate=attr)
    def new_func(self, X, y=None, **kw):
        nonlocal method
        X, y = MultiLayer._concat_features(self, X, y=y)
        func = getattr(self.estimator, method)
        if 'predict' in method:
            return func(X, **kw)
        return func(X, y=y, **kw)
    return new_func


class MultiLayer(SklearnMixin, BaseEstimator):
    __doc__ = _DOC_TEMPLATE.format(name="MultiLayer",
                                   oneliner=_hi_oneliner,
                                   description=_hi_description,
                                   parameters=_hi_parameters,
                                   example=_hi_example)

    def __init__(self, estimator, estimators=None):
        self.estimator = estimator
        self.estimators = estimators

    def _concat_features(self, X, y=None, **kw):
        X, y, row_idx = self._as_numpy_arrs(X, y)
        predicts = (getattr(est, 'predict') for est in self.estimators)
        preds = [pred(X) for pred in predicts]
        X2 = np.array(preds).T
        return X2, y

    fit = concat_features('fit')
    transform = concat_features('transform')
    fit_transform = concat_features('transform')
    predict = concat_features('predict')
    score = concat_features('score')
