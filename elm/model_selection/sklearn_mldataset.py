from functools import partial
import numpy as np
from sklearn.base import BaseEstimator
from dask.utils import derived_from # May be useful here?
from sklearn.utils.metaestimators import if_delegate_has_method # May be useful here?
from sklearn.linear_model import LinearRegression as skLinearRegression
from xarray_filters.mldataset import MLDataset
import xarray as xr


def to_np(method, cls=None):
    def new_func(self, X, y=None, **kw):
        nonlocal method
        nonlocal cls
        _cls = cls or getattr(self, '_cls', None)
        if _cls is None:
            raise ValueError('Define .cls as a scikit-learn estimator')
        func = getattr(_cls, method, None)
        if func is None:
            raise ValueError('{} is not an attribute of {}'.format(method, _cls))
        if hasattr(X, 'to_array'):
            X, y = X.to_array(y=y)
        if 'predict' in method:
            return func(self, X, **kw)
        return func(self, X, y=y, **kw)

    return new_func


class SklearnMixin:
    _cls = None
    def _to_np(self, X, y=None, **kw) :
        if isinstance(X, xr.Dataset):
            X = MLDataset(X)
        if hasattr(X, 'has_features'):
            if X.has_features(raise_err=False):
                pass
            else:
                X = X.to_features()
        if hasattr(X, 'astype') and not isinstance(X, np.ndarray):
            X = X.astype('numpy')
        # TODO - if y is not numpy array, then the above lines are needed for y
        return X, y

    def _call_np_method(self, method, X, y=None, **kw):
        X, y = self._to_np(X, y)
        return getattr(self, method)(X, y, **kw)

    fit_transform = to_np('fit_transform')
    fit = to_np('fit')
    predict = to_np('predict')
    # TODO -others ....


class LinearRegression(SklearnMixin, skLinearRegression):
    _cls = skLinearRegression
    def __init__(self, fit_intercept=True, normalize=False,
                 copy_X=True, n_jobs=1):
        skLinearRegression.__init__(self,
                                    fit_intercept=fit_intercept,
                                    normalize=normalize,
                                    copy_X=copy_X,
                                    n_jobs=n_jobs)

'''def loop_sklearn():
    import sklearn as sk
    for attr in dir(sk):
        if '__' == attr[0]:
            continue
        module = __import__('sklearn.{}'.format(attr))
        for item'''

mod = LinearRegression()
from xarray_filters.tests.test_data import new_test_dataset
X = new_test_dataset(('a', 'b', 'c'))
f = X.to_features()
y = f.features.values.dot(np.random.uniform(0, 1, f.features.layer.size))
mod.fit(X, y)
mod.predict(X)



