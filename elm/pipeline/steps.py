import copy
import numpy as np
import xarray as xr

from elm.pipeline.preproc_scale import *
from elm.pipeline.step_mixin import StepMixin
from elm.pipeline.transform import Transform
from elm.readers import ElmStore
from elm.config import ElmConfigError
from elm.model_selection.util import get_args_kwargs_defaults
from elm.sample_util import encoding_scaling as elmpre
from elm.sample_util.feature_selection import feature_selection_base
from elm.sample_util.sample_pipeline import _split_pipeline_output
import elm.sample_util.change_coords as change_coords



class ChangeCoordsMixin(StepMixin):

    _mod = change_coords

    def __init__(self, arg=None, _sp_step=None, _context=None, **kwargs):
        self._sp_step = _sp_step or self._sp_step
        self._context = _context or self._context
        self._kwargs = {self._sp_step: arg, **kwargs}
        self.__func, args, kwargs = change_coords.change_coords_dict_action(self._kwargs)
        super(ChangeCoordsMixin, self).__init__(**self._kwargs)

    def fit_transform(self, X, y=None, sample_weight=None, **kwargs):
        kw = kwargs.copy()
        kw['y'] = y
        kw['sample_weight'] = sample_weight
        args, kwargs = self._filter_kw(self.__func,X,**kw)
        out = self.__func(*args, **kwargs)
        return _split_pipeline_output(out, out, y, sample_weight,
                                      'ChangeCoordsMixin')

    fit = transform = fit_transform

    def _filter_kw(self, func, X, **kwargs):
        kw = {k: v for k, v in {**kwargs, **self._kwargs}.items()
              if k != self._sp_step}
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


