import copy
import logging

import numpy as np
import xarray as xr

from elm.sample_util.step_mixin import StepMixin
from elm.readers import ElmStore

logger = logging.getLogger(__name__)

__all__ = ['Transform',]



class Transform(StepMixin):
    def __init__(self, estimator, partial_fit_batches=None):
        self._estimator = estimator
        self._partial_fit_batches = partial_fit_batches
        self._params = estimator.get_params()

    def set_params(self, **params):
        filtered = {k: v for k, v in params.items()
                    if k != 'partial_fit_batches'}
        self._estimator.set_params(**filtered)
        self._params.update(params)
        p = params.get('partial_fit_batches')
        if p:
            self._partial_fit_batches = p

    def get_params(self, **kwargs):
        params = self._estimator.get_params(**kwargs)
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
                raise ValueError("Call elm.pipeline.steps.Flatten() before Transform in pipeline or otherwise use X as an (elm.readers.ElmStore or xarray.Dataset)")
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
            attrs.update(X.flat.attrs)
            attrs['band_order'] = band
            Xnew = ElmStore({'flat': xr.DataArray(out,
                            coords=coords,
                            dims=X.flat.dims,
                            attrs=attrs)},
                        attrs=attrs)
            return (Xnew, y, sample_weight)
        return out # a fitted "self"

    def partial_fit_batches(self, X, y=None, sample_weight=None, **kwargs):
        for _ in range(self._partial_fit_batches):
            logger.debug('Transform partial fit batch {} of {}'.format(_ + 1, self._partial_fit_batches))
            self.partial_fit(X, y=y, sample_weight=sample_weight, **kwargs)
        return self

    def partial_fit(self, X, y=None, sample_weight=None, **kwargs):
        if not hasattr(self._estimator, 'partial_fit'):
            raise ValueError('Cannot give partial_fit_batches to {} (does not have "partial_fit" method)'.format(self._estimator))
        return self._fit_trans('partial_fit', X, y=y, sample_weight=sample_weight, **kwargs)

    def fit(self, X, y=None, sample_weight=None, **kwargs):
        if self._partial_fit_batches:
            return self.partial_fit_batches(X, y=y, sample_weight=sample_weight, **kwargs)
        return self._fit_trans('fit', X, y=y, sample_weight=sample_weight, **kwargs)

    def transform(self, X, y=None, sample_weight=None, **kwargs):
        return self._fit_trans('transform', X, y=y, sample_weight=sample_weight, **kwargs)

    def fit_transform(self, X, y=None, sample_weight=None, **kwargs):
        fitted = self.fit(X, y=y, sample_weight=sample_weight, **kwargs)
        return self.transform(X, y=y, sample_weight=sample_weight, **kwargs)


