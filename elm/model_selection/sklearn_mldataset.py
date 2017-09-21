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


def to_np(method, cls=None):
    def new_func(self, X, y=None, **kw):
        nonlocal method
        nonlocal cls
        _cls = cls or getattr(self, '_cls', None)
        if _cls is None:
            raise ValueError('Define .cls as a scikit-learn estimator')
        func = getattr(_cls, method, None)
        if _cls.__name__ == 'Pipeline':
            func = getattr(self._final_estimator, method, None)
        a, k, defaults = get_args_kwargs_defaults(func)
        print('a',a,k)
        print('self', self, cls)
        if func is None:
            raise ValueError('{} is not an attribute of {}'.format(method, _cls))
        if hasattr(X, 'to_array'):
            X, y = X.to_array(y=y)
        args = (self, X)
        if not 'self' in a:
            args = args[1:]
        print('ags', len(args))
        if 'predict' in method:
            return func(*args)
        return func(*args, y=y, **kw)
    return new_func


class SklearnMixin:
    _cls = None
    def _to_np(self, X, y=None, **kw) :
        features_layer = kw.get('features_layer', FEATURES_LAYER)
        if isinstance(X, xr.Dataset):
            X = MLDataset(X)
        if hasattr(X, 'has_features'):
            if X.has_features(raise_err=False):
                pass
            else:
                X = X.to_features()
        if isinstance(X, MLDataset):
            arr = X[features_layer]
            row_idx = getattr(arr, arr.dims[0])
        else:
            row_idx = None
        if hasattr(X, 'to_array') and not isinstance(X, np.ndarray):
            X, y = X.to_array()
        # TODO - if y is not numpy array, then the above lines are needed for y
        return X, y, row_idx

    def _from_np(self, y, row_idx, features_layer=None):
        features_layer = features_layer or FEATURES_LAYER
        coords = [row_idx,
                  (FEATURES_LAYER_DIMS[1], np.array(['predict']))]
        dims = FEATURES_LAYER_DIMS
        if y.ndim == 1:
            y = y[:, np.newaxis]
        arr = xr.DataArray(y, coords=coords, dims=dims)
        dset = MLDataset(OrderedDict([(features_layer, arr)]))
        return dset.from_features(features_layer=features_layer)

    def _predict(self, X):
        X2, _, row_idx = self._to_np(X, y=None)
        print('X2', X2.shape, X.data_vars.keys())
        y3 = self._cls.predict(self, X2)
        return self._from_np(y3, row_idx)

    fit_transform = to_np('fit_transform')
    fit = to_np('fit')
    predict = _predict


