import numpy as np

from elm.sample_util.step_mixin import StepMixin
from elm.config import ElmConfigError, import_callable
from elm.readers import (select_canvas as _select_canvas,
                         drop_na_rows as _drop_na_rows,
                         ElmStore,
                         flatten as _flatten,
                         inverse_flatten as _inverse_flatten,
                         Canvas,
                         check_is_flat,
                         transpose as _transpose,
                         aggregate_simple)

CHANGE_COORDS_ACTIONS = (
    'select_canvas',
    'flatten',
    'drop_na_rows',
    'inverse_flatten',
    'modify_sample',
    'transpose',
    'agg',
)


class SelectCanvas(StepMixin):
    _sp_step = 'select_canvas'
    def __init__(self, band=None):
        self.band = band

    def get_params(self):
        return {'band': self.band}

    def set_params(self, **params):
        if not params or (params and 'band' not in params):
            raise ValueError('Only "band" is a valid parameter to SelectCanvas')
        self.band = params['band']

    def fit_transform(self, X, y=None, sample_weight=None, **kwargs):
        band_arr = getattr(X, self.band, None)
        if band_arr is None:
            raise ValueError('Argument to select_canvas should be a band name, e.g. "band_1" (found {} but bands are {})'.format(band, X.data_vars))
        new_canvas = band_arr.canvas
        X = _select_canvas(X, new_canvas)
        return (X, y, sample_weight)

    transform = fit = fit_transform

    @classmethod
    def from_config_dict(cls, **kwargs):
        return cls(kwargs['select_canvas'])

class Flatten(StepMixin):
    _sp_step = 'flatten'

    def __init__(self):
        pass

    def fit_transform(self, X, y=None, sample_weight=None, **kwargs):
        return (_flatten(X), y, sample_weight)

    transform = fit = fit_transform

    def get_params(self):
        return {}

    def set_params(self, **params):
        if params:
            raise ValueError("Flatten takes no arguments")
    @classmethod
    def from_config_dict(cls, **kwargs):
        return cls()

class DropNaRows(StepMixin):
    _sp_step = 'drop_na_rows'
    def __init__(self):
        pass

    def fit_transform(self, X, y=None, sample_weight=None, **kwargs):
        return (_drop_na_rows(X), y, sample_weight)

    transform = fit = fit_transform

    def get_params(self):
        return {}

    def set_params(self, **params):
        if params:
            raise ValueError('DropNaRows takes no parameters')

    @classmethod
    def from_config_dict(cls, **kwargs):
        return cls()

class InverseFlatten(StepMixin):
    _sp_step = 'inverse_flatten'
    def __init__(self):
        pass

    def fit_transform(self, X, y=None, sample_weight=None, **kwargs):
        return (_inverse_flatten(X), y, sample_weight)

    transform = fit = fit_transform

    def get_params(self):
        return {}

    def set_params(self, **params):
        if params:
            raise ValueError('DropNaRows takes no parameters')

    @classmethod
    def from_config_dict(cls, **kwargs):
        return cls()

class Transpose(StepMixin):
    _sp_step = 'transpose'
    def __init__(self, trans_arg):
        self.trans_arg = trans_arg

    def fit_transform(self, X, y=None, sample_weight=None, **kwargs):
        return (_transpose(X, self.trans_arg), y, sample_weight)

    transform = fit = fit_transform

    def get_params(self):
        return {'trans_arg': self.trans_arg}

    def set_params(self, **params):
        if params and not 'trans_arg' in params:
            raise ValueError('Transpose set_params takes only "trans_arg" - argument to xarray transpose')

    @classmethod
    def from_config_dict(cls, **kwargs):
        return cls(kwargs['transpose'])

class Agg(StepMixin):
    _sp_step = 'agg'
    def __init__(self, func=None, axis=None, dim=None):
        self.axis = axis
        self.dim = dim
        if dim is None and axis is None:
            raise ValueError('Expected dim or axis in keyword __init__ args')
        if not func:
            raise ValueError('Expected "func" in keyword arguments to Agg')
        self.func = func

    def fit_transform(self, X, y=None, sample_weight=None, **kwargs):
        X = aggregate_simple(X, axis=self.axis, dim=self.dim,
                             func=self.func)
        return (X, y, sample_weight)

    transform = fit = fit_transform

    def get_params(self):
        return {'axis': self.axis, 'dim': self.dim, 'func': self.func}

    def set_params(self, **params):
        if not 'axis' in params and not 'dim' in params and not 'func' in params:
            raise ValueError('Agg requires "dim", "axis", or "func" keywords')

    @classmethod
    def from_config_dict(cls, **kwargs):
        return cls(**kwargs['agg'])


class ModifySample(StepMixin):
    _sp_step = 'modify_sample'
    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    def fit_transform(self, X, y=None, sample_weight=None, **kwargs):
        from elm.sample_util.sample_pipeline import _split_pipeline_output
        kw = dict(y=y, sample_weight=sample_weight, **kwargs)
        kw.update(self.kwargs)
        func = import_callable(self.func)
        output = func(X, **kw)
        return _split_pipeline_output(output, X, y, sample_weight, 'ModifySample')

    transform = fit = fit_transform

    def get_params(self):
        return dict(func=self.func, **self.kwargs)

    def set_params(self, **params):
        if 'func' in params:
            self.func = params['func']
        self.kwargs.update({k: v for k, v in params.items() if k != 'func'})

    @classmethod
    def from_config_dict(cls, **kwargs):
        kw = dict(kwargs)
        func = kw.pop('modify_sample')
        return cls(func, **kw)


gs = tuple(globals().items())
__all__ = []
for k,v in gs:
    if isinstance(v, type) and issubclass(v, StepMixin):
        __all__.append(k)

