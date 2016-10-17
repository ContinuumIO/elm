

from elm.pipeline.preproc_scale import *
from elm.pipeline.step_mixin import StepMixin
from elm.pipeline.transform import Transform
import elm.sample_util.change_coords as change_coords



class ChangeCoordsMixin(StepMixin):

    _mod = change_coords

    def __init__(self, arg=None, **kwargs):
        self._kwargs = {self._sp_step: arg, **kwargs}
        self.__func, args, kwargs = change_coords.change_coords_dict_action(self._kwargs)
        super(ChangeCoordsMixin, self).__init__(**self._kwargs)

    def fit_transform(self, X, y=None, sample_weight=None, **kwargs):
        from elm.sample_util.sample_pipeline import _split_pipeline_output
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
        return self._kwargs.copy()

    def set_params(self, **kw):
        self._kwargs.update(kw)


class SelectCanvas(ChangeCoordsMixin):
    _sp_step = 'select_canvas'


class Flatten(ChangeCoordsMixin):
    _sp_step = 'flatten'


class DropNaRows(ChangeCoordsMixin):
    _sp_step = 'drop_na_rows'


class InverseFlatten(ChangeCoordsMixin):
    _sp_step = 'inverse_flatten'


class Transpose(ChangeCoordsMixin):
    _sp_step = 'transpose'


class Agg(ChangeCoordsMixin):
    _sp_step = 'agg'


class ModifySample(ChangeCoordsMixin):
    _sp_step = 'modify_sample'

