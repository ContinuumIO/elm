import copy
import logging
from functools import partial

import numpy as np
import xarray as xr

from elm.pipeline.step_mixin import StepMixin
from elm.config import import_callable
from elm.pipeline.serialize import load_models_from_tag
from elm.readers import ElmStore, check_is_flat

logger = logging.getLogger(__name__)

__all__ = ['Transform',]



class Transform(StepMixin):
    def __init__(self, estimator, partial_fit_batches=None):
        self._estimator = estimator
        self._partial_fit_batches = partial_fit_batches
        self._params = estimator.get_params()

    def set_params(self, **params):
        filtered = {k: v for k, v in self._params
                    if k != 'partial_fit_batches'}
        self._estimator.set_params(**filtered)
        self._params.update(params)
        p = params.get('partial_fit_batches')
        if p:
            self._partial_fit_batches = p

    def get_params(self):
        params = self._estimator.get_params()
        params['partial_fit_batches'] = self._partial_fit_batches
        return params

    def _fit_trans(self, method, X, y=None, sample_weight=None, **kwargs):
        fitter_func = getattr(self._estimator, method)
        kw = dict(y=y, sample_weight=sample_weight, **kwargs)
        kw = {k: v for k, v in kw.items() if k in self._params}
        if isinstance(X, (ElmStore, xr.Dataset)):
            if hasattr(X, 'flat'):
                XX = X.flat.values
                space = X.flat.space
            else:
                raise ValueError("Call elm.pipeline.steps.Flatten('C') before Transform in pipeline or otherwise use X as an (elm.readers.ElmStore or xarray.Dataset)")
        else:
            raise ValueError('Expected X to be an xarray.Dataset or elm.readers.ElmStore')
        out = fitter_func(X.flat.values, **kw)
        if 'transform' in method:
            # 'transform' or 'fit_transform' was called
            out = np.atleast_2d(out)
            band = ['transform_{}'.format(idx)
                    for idx in range(out.shape[1])]
            coords = [('space', space),
                      ('band', band)]
            attrs = copy.deepcopy(X.attrs)
            attrs['band_order'] = band
            Xnew = ElmStore({'flat': xr.DataArray(out,
                            coords=coords,
                            dims=X.dims,
                            attrs=attrs)},
                        attrs=attrs)
            return (Xnew, y, sample_weight)
        return out # a fitted "self"

    def partial_fit_batches(self, X, y=None, sample_weight=None, **kwargs):
        fitted = self
        for _ in range(self._partial_fit_batches):
            fitted = self.partial_fit(X, y=y, sample_weight=sample_weight, **kwargs)
        return self

    def partial_fit(self, X, y=None, sample_weight=None, **kwargs):
        if not hasattr(self._estimator, 'partial_fit'):
            raise ValueError('Cannot give partial_fit_batches to {} (does not have "partial_fit" method)'.format(self._estimator))
        return self._fit_trans('transform', X, y=y, sample_weight=sample_weight, **kwargs)

    def fit(self, X, y=None, sample_weight=None, **kwargs):
        if self._partial_fit_batches:
            return self.partial_fit_batches(X, y=y, sample_weight=sample_weight, **kwargs)
        return self._fit_trans('fit', X, y=y, sample_weight=sample_weight, **kwargs)

    def transform(self, X, y=None, sample_weight=None, **kwargs):
        return self._fit_trans('transform', X, y=y, sample_weight=sample_weight, **kwargs)

    def fit_transform(self, X, y=None, sample_weight=None, **kwargs):
        if hasattr(self._estimator, 'fit_transform'):
            return self._fit_trans('fit_transform', X, y=y,
                                   sample_weight=sample_weight, **kwargs)
        fitted = self._fit_trans('fit', X, y=y,
                        sample_weight=sample_weight, **kwargs)
        return fitted.transform(X)



def _get_saved_transform_models(action, config, **kwargs):
    method = action.get('method', 'fit_transform')
    tag = action['transform']
    logger.debug('Transform method does not include "fit"')
    logger.info('Load pickled transform_models from {} {}'.format(config.ELM_TRANSFORM_PATH, tag))
    transform_models, meta = load_models_from_tag(config.ELM_TRANSFORM_PATH, tag)
    return transform_models


def init_saved_transform_models(config, pipeline):

    transform_model = None
    for action in pipeline:
        if 'transform' in action:
            transform = copy.deepcopy(config.transform[action['transform']])
            transform_model = _get_saved_transform_models(action,
                                                          config,
                                                          **transform)
    logger.debug('Initialized transform model {}'.format(transform_model))
    return transform_model


def get_new_or_saved_transform_model(config, pipeline, data_source, step):
    transform_model = None
    train_or_transform = 'train' if 'train' in step else 'transform'
    for item in pipeline:
        if 'transform' in item:
            method = item.get('method', config.transform.get('method', None))
            if method is None:
                raise ValueError('Expected a "method" for transform')
            if 'fit' not in method:
                return init_saved_transform_models(config, pipeline)
            else:
                model_args = _make_model_args_from_config(config,
                                                          config.transform[item['transform']],
                                                          step,
                                                          train_or_transform,
                                                          pipeline,
                                                          data_source)
                model = model_args.model_init_class(**model_args.model_init_kwargs)
                return [('tag_0', model)]
    return None
