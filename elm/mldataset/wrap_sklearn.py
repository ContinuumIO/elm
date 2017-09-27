from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
from functools import partial
from importlib import import_module
import os

import numpy as np
from sklearn.base import BaseEstimator
from dask.utils import derived_from # May be useful here?
from sklearn.utils.metaestimators import if_delegate_has_method # May be useful here?
from sklearn.linear_model import LinearRegression as skLinearRegression
from xarray_filters.mldataset import MLDataset
from xarray_filters.func_signatures import filter_args_kwargs
from xarray_filters.constants import FEATURES_LAYER_DIMS, FEATURES_LAYER
import xarray as xr
import yaml


def get_row_index(X, features_layer=None):
    if features_layer is None:
        features_layer = FEATURES_LAYER
    if isinstance(X, MLDataset):
        arr = X[features_layer]
        return getattr(arr, arr.dims[0])

def _as_numpy_arrs(self, X, y=None, **kw) :
    if isinstance(X, xr.Dataset):
        X = MLDataset(X)
    if hasattr(X, 'has_features'):
        if X.has_features(raise_err=False):
            pass
        else:
            X = X.to_features()
    row_idx = get_row_index(X)
    if hasattr(X, 'to_array') and not isinstance(X, np.ndarray):
        X, y = X.to_array(y=y)
        # TODO what about row_idx now?
    # TODO - if y is not numpy array, then the above lines are needed for y
    return X, y, row_idx


def _from_numpy_arrs(self, y, row_idx, features_layer=None):
    if isinstance(y, MLDataset):
        return y
    features_layer = features_layer or FEATURES_LAYER
    coords = [row_idx,
              (FEATURES_LAYER_DIMS[1], np.array(['predict']))]
    dims = FEATURES_LAYER_DIMS
    if y.ndim == 1:
        y = y[:, np.newaxis]
    arr = xr.DataArray(y, coords=coords, dims=dims)
    dset = MLDataset(OrderedDict([(features_layer, arr)]))
    return dset.from_features(features_layer=features_layer)


class SklearnMixin:
    _cls = None
    _as_numpy_arrs = _as_numpy_arrs
    _from_numpy_arrs = _from_numpy_arrs

    def _call_sk_method(self, sk_method, X, y=None, **kw):
        _cls = self._cls
        if _cls is None:
            raise ValueError('Define .cls as a scikit-learn estimator')
        func = getattr(_cls, sk_method, None)
        if func is None:
            raise ValueError('{} is not an attribute of {}'.format(sk_method, _cls))
        X, y, _ = self._as_numpy_arrs(X, y=y)
        kw.update(dict(self=self, X=X))
        if y is not None:
            kw['y'] = y
        kw = filter_args_kwargs(func, **kw)
        return func(**kw)

    def _predict_steps(self, X, row_idx=None, sk_method=None, **kw):
        X2, _, temp_row_idx = self._as_numpy_arrs(X, y=None)
        if temp_row_idx is not None:
            row_idx = temp_row_idx
        y3 = self._call_sk_method(sk_method, X2, **kw)
        return y3, row_idx

    def predict(self, X, row_idx=None, **kw):
        y, row_idx = self._predict_steps(X, row_idx=row_idx,
                                         sk_method='predict', **kw)
        return self._from_numpy_arrs(y, row_idx)

    def predict_proba(self, X, row_idx=None, **kw):
        proba, row_idx = self._predict_steps(X, row_idx=row_idx,
                                             sk_method='predict_proba', **kw)
        return proba

    def predict_log_proba(self, X, row_idx=None, **kw):
        log_proba, row_idx = self._predict_steps(X, row_idx=row_idx,
                                                 sk_method='predict_log_proba',
                                                 **kw)
        return log_proba

    def decision_function(self, X, row_idx=None, **kw):
        d, row_idx = self._predict_steps(X, row_idx=row_idx,
                                         sk_method='decision_function',
                                         **kw)
        return d

    def fit(self, X, y=None, **kw):
        self._call_sk_method('fit', X, y=y, **kw)
        return self

    def _fit(self, X, y=None, **kw):
        return self._call_sk_method('_fit', X, y=y, **kw)

    def transform(self, X, y=None, **kw):
        if hasattr(self._cls, 'transform'):
            return self._call_sk_method('transform', X, y=y, **kw)
        if hasattr(self._cls, 'fit_transform'):
            return self._call_sk_method('fit_transform', X, y=y, **kw)
        raise ValueError('Estimator {} has no "transform" or '
                         '"fit_transform" methods'.format(self))

    def fit_transform(self, X, y=None, **kw):
        args = (X,)
        if y is not None:
            args = args + (y,)
        if hasattr(self._cls, 'fit_transform'):
            return self._call_sk_method('fit_transform', *args, **kw)
        self.fit(*args, **kw)
        return self._call_sk_method('transform', *args, **kw)

    def __repr__(self):
        return self._cls.__repr__(self)

    def __str__(self):
        return self._cls.__str__(self)

    def fit_predict(self, X, y=None, **kw):
        return self.fit(X, y=y, **kw).predict(X)

