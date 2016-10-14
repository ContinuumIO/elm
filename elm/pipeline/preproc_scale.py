import copy
from functools import WRAPPER_ASSIGNMENTS, wraps, partial

import sklearn.feature_selection as skfeat
import sklearn.preprocessing as skpre
import numpy as np
import xarray as xr

from elm.model_selection.util import get_args_kwargs_defaults
from elm.readers import *
from elm.pipeline.step_mixin import StepMixin

class SklearnBase(StepMixin):
    _sp_step = None
    _method = None
    _mod = None
    def __init__(self, _method=None, _mod=None, **kwargs):
        cls = _method or self._method # get the sklearn class
        if _mod:
            self._mod = _mod
        self._estimator = cls(**kwargs) # init on that class

    def require_flat(self, X):
        if not isinstance(X, (ElmStore, xr.Dataset)) and hasattr(X, 'flat'):
            raise ValueError("Expected an elm.readers.ElmStore or xarray.Dataset with DataArray 'flat' (2-d array with dims [space, band])")

    def _filter_kw(self, func, X, y=None, sample_weight=None, **kwargs):
        args, defaults, var_kwargs = get_args_kwargs_defaults(func)
        kw = dict(y=y, sample_weight=sample_weight, **kwargs)
        kw = {k: v for k, v in kw.items() if k in defaults or k in args}
        return ((X.flat.values,), kw, y, sample_weight)

    def get_params(self):
        return self._estimator.get_params()

    def set_params(self, **params):
        return self._estimator.set_params(**params)

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
        return self._estimator.transform(*args, **kwargs)

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


class SklearnPreproc(SklearnBase):
    _sp_step = 'sklearn_preprocessing'
    _mod = skpre



class SklearnFeatSelect(SklearnBase):
    _sp_step = 'feature_selection'
    _mod = skfeat


class Binarizer(SklearnPreproc):
    _context = "Binarizer"
    _method = skpre.Binarizer


class FunctionTransformer(SklearnPreproc):
    _context = "FunctionTransformer"
    _method = skpre.FunctionTransformer


class Imputer(SklearnPreproc):
    _context = "Imputer"
    _method = skpre.Imputer


class KernelCenterer(SklearnPreproc):
    _context = "KernelCenterer"
    _method = skpre.KernelCenterer


class LabelBinarizer(SklearnPreproc):
    _context = "LabelBinarizer"
    _method = skpre.LabelBinarizer


class LabelEncoder(SklearnPreproc):
    _context = "LabelEncoder"
    _method = skpre.LabelEncoder


class MaxAbsScaler(SklearnPreproc):
    _context = "MaxAbsScaler"
    _method = skpre.MaxAbsScaler


class MinMaxScaler(SklearnPreproc):
    _context = "MinMaxScaler"
    _method = skpre.MinMaxScaler


class MultiLabelBinarizer(SklearnPreproc):
    _context = "MultiLabelBinarizer"
    _method = skpre.MultiLabelBinarizer


class Normalizer(SklearnPreproc):
    _context = "Normalizer"
    _method = skpre.Normalizer


class OneHotEncoder(SklearnPreproc):
    _context = "OneHotEncoder"
    _method = skpre.OneHotEncoder


class PolynomialFeatures(SklearnPreproc):
    _context = "PolynomialFeatures"
    _method = skpre.PolynomialFeatures


class RobustScaler(SklearnPreproc):
    _context = "RobustScaler"
    _method = skpre.RobustScaler


class StandardScaler(SklearnPreproc):
    _context = "StandardScaler"
    _method = skpre.StandardScaler

class GenericUnivariateSelect(SklearnFeatSelect):
    _context = "GenericUnivariateSelect"
    _method = skfeat.GenericUnivariateSelect


class RFE(SklearnFeatSelect):
    _context = "RFE"
    _method = skfeat.RFE


class RFECV(SklearnFeatSelect):
    _context = "RFECV"
    _method = skfeat.RFECV


class SelectFdr(SklearnFeatSelect):
    _context = "SelectFdr"
    _method = skfeat.SelectFdr


class SelectFpr(SklearnFeatSelect):
    _context = "SelectFpr"
    _method = skfeat.SelectFpr


class SelectFromModel(SklearnFeatSelect):
    _context = "SelectFromModel"
    _method = skfeat.SelectFromModel


class SelectFwe(SklearnFeatSelect):
    _context = "SelectFwe"
    _method = skfeat.SelectFwe


class SelectKBest(SklearnFeatSelect):
    _context = "SelectKBest"
    _method = skfeat.SelectKBest


class SelectPercentile(SklearnFeatSelect):
    _context = "SelectPercentile"
    _method = skfeat.SelectPercentile


class VarianceThreshold(SklearnFeatSelect):
    _context = "VarianceThreshold"
    _method = skfeat.VarianceThreshold

def _set_wrapper_info(cls):
    orig = cls._method
    for method in ('fit_transform', '__init__', 'transform', None):
        if method:
            m = getattr(orig, method, None)
        else:
            m = orig
        if not m:
            continue
        for at in tuple(WRAPPER_ASSIGNMENTS) + ('__repr__', '__str__'):
            content = getattr(m, at, None)
            if content:
                func = getattr(cls, method, None) if method else cls
                if func:
                    setattr(func, at, content)

gs = tuple(globals().items())
clses = [v for k, v in gs if isinstance(v, type) and issubclass(v, SklearnBase)]
for cls in clses:
    _set_wrapper_info(cls)

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