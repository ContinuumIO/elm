import copy
from functools import WRAPPER_ASSIGNMENTS, wraps, partial

import sklearn.feature_selection as skfeat
import sklearn.preprocessing as skpre
import numpy as np
import xarray as xr

from elm.config import import_callable
from elm.model_selection.util import get_args_kwargs_defaults
from elm.readers import *
from elm.pipeline.step_mixin import StepMixin

class SklearnBase(StepMixin):
    def __init__(self,  **kwargs):
        cls = getattr(skfeat, self.__class__.__name__, None)
        if cls is None:
            cls = getattr(skpre, self.__class__.__name__)
        kwargs = self._import_score_func(**kwargs)
        args, defaults, var_kwargs = get_args_kwargs_defaults(cls.__init__)
        kw = {k: v for k, v in kwargs.items() if k in defaults or k in args}
        self._estimator = cls(**kw) # init on that class

    def _import_score_func(self, **params):
        if 'score_func' in params:
            if isinstance(params['score_func'], str):
                sf = getattr(skfeat, params['score_func'], None)
                if not sf:
                    sf = import_callable(params['score_func'])
                params['score_func'] = sf
        return params

    def require_flat(self, X):
        if not (isinstance(X, (ElmStore, xr.Dataset)) and hasattr(X, 'flat')):
            raise ValueError("Expected an elm.readers.ElmStore or xarray.Dataset with DataArray 'flat' (2-d array with dims [space, band])")

    def _filter_kw(self, func, X, y=None, sample_weight=None, **kwargs):
        args, defaults, var_kwargs = get_args_kwargs_defaults(func)
        kw = dict(y=y, sample_weight=sample_weight, **kwargs)
        kw = {k: v for k, v in kw.items() if k in defaults or k in args}
        return ((X.flat.values,), kw, y, sample_weight)

    def get_params(self, **kwargs):
        params = self._estimator.get_params()
        return self._estimator.get_params(**kwargs)

    def set_params(self, **params):
        kwargs = self._import_score_func(**params)
        kwargs = {k: v for k, v in params.items()
                  if k in self._estimator.get_params()}
        self._estimator.set_params(**kwargs)
        self_kwargs = {k: v for k, v in params.items()
                       if k not in kwargs}
        for k, v in self_kwargs.items():
            setattr(self, k, v)

    def transform(self, *args, **kwargs):
        X = args[0]
        self.require_flat(X)
        args, kwargs, y, sample_weight = self._filter_kw(self._estimator.transform, *args, **kwargs)
        new_X = self._estimator.transform(*args, **kwargs)
        X = self._to_elm_store(new_X, X)
        return (X, y, sample_weight)

    def fit_transform(self, *args, **kwargs):
        X = args[0]
        if hasattr(self._estimator, 'fit_transform'):
            self.require_flat(X)
            args, kwargs, y, sample_weight = self._filter_kw(self._estimator.fit_transform, *args, **kwargs)
            new_X = self._estimator.fit_transform(*args, **kwargs)
            X = self._to_elm_store(new_X, X)
            return (X, y, sample_weight)
        self._estimator = self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)

    def fit(self, *args, **kwargs):
        X = args[0]
        self.require_flat(X)
        args, kwargs, _, _ = self._filter_kw(self._estimator.fit, *args, **kwargs)
        return self._estimator.fit(*args, **kwargs)

    def _to_elm_store(self, X, old_X):
        attrs = copy.deepcopy(old_X.attrs)
        attrs.update(copy.deepcopy(old_X.flat.attrs))
        band = ['feat_{}'.format(idx) for idx in range(X.shape[1])]
        flat = xr.DataArray(X,
                            coords=[('space', old_X.flat.space), ('band', band)],
                            dims=old_X.flat.dims,
                            attrs=attrs)
        return ElmStore({'flat': flat}, attrs=attrs)


class Binarizer(SklearnBase):
    pass

class FunctionTransformer(SklearnBase):
    pass

class Imputer(SklearnBase):
    pass

class KernelCenterer(SklearnBase):
    pass

class LabelBinarizer(SklearnBase):
    pass

class LabelEncoder(SklearnBase):
    pass

class MaxAbsScaler(SklearnBase):
    pass

class MinMaxScaler(SklearnBase):
    pass

class MultiLabelBinarizer(SklearnBase):
    pass

class Normalizer(SklearnBase):
    pass

class OneHotEncoder(SklearnBase):
    pass

class PolynomialFeatures(SklearnBase):
    pass

class RobustScaler(SklearnBase):
    pass

class StandardScaler(SklearnBase):
    pass

class RFE(SklearnBase):
    pass

class RFECV(SklearnBase):
    pass

class SelectFdr(SklearnBase):
    pass

class SelectFpr(SklearnBase):
    pass

class SelectFromModel(SklearnBase):
    pass

class SelectFwe(SklearnBase):
    pass

class SelectKBest(SklearnBase):
    pass

class SelectPercentile(SklearnBase):
    pass

class VarianceThreshold(SklearnBase):
    pass


gs = tuple(globals().items())
clses = [(k, v) for k, v in gs if isinstance(v, type) and issubclass(v, SklearnBase)]
SKLEARN_PREPROCESSING = {}
for k, cls in clses:
    SKLEARN_PREPROCESSING[k] = cls

def require_positive(X, small_num=0.0001):
    '''Helper function to ensure positivity before functions like "log"
    Params:
        X:  numpy array
        small_num: small float number which should replace values <= 0'''
    if X.dtype.kind != 'f':
        X = X.astype(np.float32)
    X[np.where(X <= 0)] = small_num
    return X

__all__ = [k for k,v in gs if k[0].isupper() and k[0] != '_']