from collections import OrderedDict
from functools import partial
import numpy as np
from sklearn.base import BaseEstimator
from dask.utils import derived_from # May be useful here?
from sklearn.utils.metaestimators import if_delegate_has_method # May be useful here?
from sklearn.linear_model import LinearRegression as skLinearRegression
from xarray_filters.mldataset import MLDataset
from xarray_filters.func_signatures import get_args_kwargs_defaults
from xarray_filters.constants import FEATURES_LAYER_DIMS, FEATURES_LAYER
import xarray as xr


def _call_sk_method(method, cls=None):
    def new_func(self, X, y=None, **kw):
        nonlocal method
        nonlocal cls
        _cls = cls or getattr(self, '_cls', None)
        if _cls is None:
            raise ValueError('Define .cls as a scikit-learn estimator')
        func = getattr(_cls, method, None)
        #if _cls.__name__ == 'Pipeline':
         #   func = getattr(self._final_estimator.__class__, method, None)
        if func is None:
            raise ValueError('{} is not an attribute of {}'.format(method, _cls))
        X, y, _ = self._as_numpy_arrs(X, y=y)

        args = (self, X)
        arg_spec, kw_spec, var_key = get_args_kwargs_defaults(func)
        print(arg_spec, kw_spec, var_key, func, 'self', self, X, y)
        if 'predict' in method:
            return func(*args)
        if y is not None:
            args = args + (y,)
        return func(*args, **kw)
    return new_func

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

    def predict(self, X):
        X2, _, row_idx = self._as_numpy_arrs(X, y=None)
        y3 = _call_sk_method('predict', cls=self._cls)(self, X2)
        return self._from_numpy_arrs(y3, row_idx)

    def fit(self, *args, **kw):
        _call_sk_method('fit', cls=self._cls)(self, *args, **kw)
        return self

    def _fit(self, *args, **kw):
        if not hasattr(self._cls, '_fit'):
            raise ValueError('')
        return _call_sk_method('_fit', cls=self._cls)(self, *args, **kw)

    transform = _call_sk_method('transform')

    def fit_transform(self, X, y=None, **kw):
        args = (self, X)
        if y is not None:
            args = args + (y,)
        #if hasattr(self._cls, 'fit_transform'):
        #    return _call_sk_method('fit_transform', cls=self._cls)(self, *args, **kw)
        _call_sk_method('fit', cls=self._cls)(self, *args, **kw)
        return self.transform(self, *args, **kw)

    def __repr__(self):
        return self._cls.__repr__(self)

    def __str__(self):
        return self._cls.__str__(self)


