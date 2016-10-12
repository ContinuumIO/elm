from functools import WRAPPER_ASSIGNMENTS, wraps, partial

import sklearn.feature_selection as skfeat
import sklearn.preprocessing as skpre

from elm.config import ElmConfigError
from elm.sample_util import encoding_scaling as elmpre
from elm.sample_util.feature_selection import feature_selection_base
from elm.sample_util.sample_pipeline import _split_pipeline_output
import elm.sample_util.change_coords as change_coords

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


class StepMixin(object):

    _sp_step = None
    _func = None
    _required_kwargs = None
    _context = 'sample pipeline step'

    def __init__(self, *args, func=None, **kwargs):
        self._args = args
        self._kwargs = kwargs

        self.func = (func or self._func)
        if self.func:
            self.func = self.func(*args, **kwargs)
        self._validate_init_base()
        if callable(getattr(self, '_validate_init', None)):
            self._validate_init(*self._args, **self._kwargs)

    def get_params(self, deep=True):
        func = getattr(self.func, 'get_params', None)
        if func:
            return func(deep=deep)
        return self._kwargs

    def _validate_init_base(self):
        if self._sp_step is None:
            raise ValueError('Expected inheriting class to define _sp_step')

    _filter_kw = None

    def fit_transform(self, X, y=None, sample_weight=None, **kwargs):
        ft = getattr(self.func, 'fit_transform', None)
        dot_transform = False
        if not callable(ft):
            ft = getattr(self.func, 'fit', None)
            if ft is not None:
                dot_transform = True
            else:
                ft = self.func
        kw = kwargs.copy()
        if sample_weight is not None:
            kw['sample_weight'] = sample_weight
        if callable(self._filter_kw):
            args, kwargs = self._filter_kw(ft, X, y=y, **kw)
        else:
            args, kwargs = (X, y), kw
        output = ft(*args, **kwargs)
        if dot_transform:
            output = output.transform(X.flat.values)
        return _split_pipeline_output(output, X, y, sample_weight,
                                      getattr(self, '_context', ft))


class ChangeCoordsMixin(StepMixin):

    _mod = change_coords

    def __init__(self, arg=None, **kwargs):
        self._kwargs = {self._sp_step: (arg or self._arg), **kwargs}
        self.__func, args, kwargs = change_coords.change_coords_dict_action(self._kwargs)
        super(ChangeCoordsMixin, self).__init__(**self._kwargs)

    def fit_transform(self, X, y=None, sample_weight=None, **kwargs):
        kw = kwargs.copy()
        kw['y'] = y
        kw['sample_weight'] = sample_weight
        args, kwargs = self._filter_kw(self.__func,X,y=y,sample_weight=sample_weight, **kwargs)
        out = self.__func(*args, **kwargs)
        return _split_pipeline_output(out, out, y, sample_weight,
                                      'ChangeCoordsMixin')

    fit = transform = fit_transform

    def _filter_kw(self, func, X, y=None, sample_weight=None, **kwargs):
        kw = {k: v for k, v in {**kwargs, **self._kwargs}.items() if k != self._sp_step}
        return (X, self._sp_step, self._kwargs[self._sp_step]), kw

    def get_params(self, deep=True):
        return self._kwargs

    def set_params(self, **kw):
        self._kwargs.update(kw)


class SelectCanvas(ChangeCoordsMixin):
    _sp_step = 'select_canvas'
    _context = 'select canvas'


class Flatten(ChangeCoordsMixin):
    _sp_step = 'flatten'
    _context = 'flatten'


class DropNaRows(ChangeCoordsMixin):
    _sp_step = 'drop_na_rows'
    _context = 'drop NA rows'


class InverseFlatten(ChangeCoordsMixin):
    _sp_step = 'inverse_flatten'
    _context = 'inverse flatten'


class Transpose(ChangeCoordsMixin):
    _sp_step = 'transpose'
    _context = 'transpose'


class Agg(ChangeCoordsMixin):
    _sp_step = 'agg'
    _context = 'agg'


class ModifySample(ChangeCoordsMixin):
    _sp_step = 'modify_sample'
    _context = 'modify sample'


class SklearnBase(StepMixin):
    _sp_step = None
    _method = None
    _mod = None
    def __init__(self, **kwargs):
        self._func = self._method or kwargs['method']
        if not self._func:
            raise ValueError('Define _method in inheriting class')
        self.kw = {k: v for k, v in kwargs.items() if k != 'method'}
        super(SklearnBase, self).__init__(**self.kw)

    def fit_transform(self, X, y=None, sample_weight=None, **kwargs):

        if self._mod == skpre:
            f = getattr(self._mod, self._context)
            out = elmpre.sklearn_preprocessing(X, f, **kwargs)
        else:
            kw = dict(selection=getattr(self._mod, self._context), **kwargs)
            out = feature_selection_base(X, y=y, **kw)
        return _split_pipeline_output(out, out, y, sample_weight, 'SklearnBase')

    def set_params(self, **kw):
        self.kw.update(kw)
        super(SklearnBase, self).__init__(**self.kw)

    def get_params(self, deep=True):
        return self.kw

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

gs = tuple(globals().items())
clses = [v for k, v in gs if isinstance(v, type) and issubclass(v, SklearnPreproc)]
for cls in clses:
    _set_wrapper_info(cls)
