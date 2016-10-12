from collections import Sequence
from functools import partial, wraps
import copy

import numpy as np
import xarray as xr

from elm.model_selection import get_args_kwargs_defaults
from elm.readers import ElmStore
from elm.pipeline import (predict_many,
                          steps as STEPS,
                          ensemble)
from elm.sample_util.sample_pipeline import (final_on_sample_step,
                                             _split_pipeline_output,
                                             create_sample_from_data_source)



class Pipeline(object):

    def __init__(self, steps):
        self.steps = steps
        self._validate_steps()
        self._names = [_[0] for _ in self.steps]

    def _run_steps(self, X=None, y=None,
                  sample_weight=None,
                  sampler=None, args_gen=None,
                  sklearn_method='fit',
                  iter_offset=0, config=None,
                  method_kwargs=None,
                  **new_params):

        X, y, sample_weight = _split_pipeline_output(X, X, y,
                                                   sample_weight,
                                                   'run_steps_once')
        if all(_ is None for _ in (X, y, sample_weight)):
            kw = dict(sampler=sampler, args_gen=args_gen, y=y, sample_weight=sample_weight)
            X, y, sample_weight = self.create_sample(config=config, **kw)
        if new_params:
            pipe = copy.deepcopy(self)
            pipe.set_params(**new_params)
        else:
            pipe = copy.deepcopy(self)
        fit_func = None
        for idx, (_, step_cls) in enumerate(pipe.steps[:-1]):
            fit_func = step_cls.fit_transform
            func_out = fit_func(X, y=y, sample_weight=sample_weight)
            if func_out is not None:
                X, y, sample_weight = _split_pipeline_output(func_out, X, y,
                                                       sample_weight, repr(fit_func))
        if fit_func and not isinstance(X, (ElmStore, xr.Dataset)):
            raise ValueError('Expected the return value of {} to be an '
                             'elm.readers:ElmStore'.format(fit_func))
        fitter_or_predict = getattr(self._estimator, sklearn_method, None)
        if fitter_or_predict is None:
            raise ValueError('Final estimator in Pipeline {} has no method {}'.format(self._estimator, sklearn_method))
        if not isinstance(self._estimator, STEPS.StepMixin):
            args, kwargs = self._post_run_pipeline(fitter_or_predict,
                                                   self._estimator,
                                                   X,
                                                   iter_offset=iter_offset,
                                                   y=y,
                                                   sample_weight=sample_weight)


        else:
            kwargs = {'y': y, 'sample_weight': sample_weight}
            args = (X,)
        kwargs.update(method_kwargs or {})
        output = fitter_or_predict(*args, **kwargs)
        if 'predict' == sklearn_method:
            return output
        if sklearn_method == 'fit':
            return self
        # transform or fit_transform most likely
        return _split_pipeline_output(output, X, y, sample_weight, 'fit_transform')

    def _post_run_pipeline(self, fitter_or_predict, estimator,
                           X, iter_offset=None,
                           y=None, sample_weight=None,
                           classes=None, **kwargs_to_method):

        return final_on_sample_step(fitter_or_predict,
                         estimator, X,
                         iter_offset,
                         kwargs_to_method,
                         y=y,
                         sample_weight=sample_weight,
                         classes=classes,
                         require_flat=True,
                      )

    def create_sample(self, config=None, **data_source):
        X = data_source.get("X", None)
        y = data_source.get('y', None)
        sample_weight = data_source.get('sample_weight', None)
        if not ('sampler' in data_source or 'args_gen' in data_source):
            if not any(data_source.get(_, None) is not None for _ in ('X', 'y', 'sample_weight')):
                raise ValueError('Expected "sampler" or "args_gen" in "data_source" or X, y, and/or sample_weight')
        if data_source.get('sampler'):
            func, args, kwargs = create_sample_from_data_source(config=config, **data_source)
            output = func(*args, **kwargs)
        else:
            output = (X, y, sample_weight)
        return _split_pipeline_output(output, X=X, y=y,
                           sample_weight=sample_weight,
                           context=getattr(self, '_context', repr(data_source)))

    def _validate_steps(self):
        steps = []
        for idx, s in enumerate(self.steps):
            if not isinstance(s, Sequence):
                name, callable_ = 'step_{}'.format(idx), s
            else:
                name, callable_ = s
            steps.append((name, callable_))
            if idx < len(self.steps) - 1:
                if not hasattr(callable_, 'fit_transform') and not (hasattr(callable_, 'fit') and hasattr(callable_, 'transform')):
                    raise ValueError("{} has no attribute 'fit_transform', or 'fit' and 'transform'".format(callable_))
            else:
                if not any(hasattr(callable_, m) for m in ('fit', 'partial_fit', 'fit_transform')):
                    raise ValueError('')
        self.steps = steps
        self._estimator = self.steps[-1][-1]
        self._transformers = self.steps[:-1]

    def get_params(self):
        params = {}
        for name, estimator in self.steps:
            for k, v in estimator.get_params().items():
                params['{}__{}'.format(name, k)] = v
        return params

    def _validate_key(self, k, context=''):
        if not k in self._names:
            raise ValueError('{} parameter - {} is not in {}'.format(context, k, self._names))

    def _split_key(self, k):
        parts = k.split('__')
        if len(parts) == 1:
            step_key, param_key = (self.steps[-1][0], parts[0])
        elif len(parts) != 2:
            raise ValueError('Cannot parse key: {} (expected one "__" token)'.format(k))
        else:
            step_key, param_key = parts
        self._validate_key(step_key)
        step = self.steps[self._names.index(step_key)]
        _, estimator = step
        if not param_key in estimator.get_params():
            raise ValueError('Parameter {} is not a keyword to {}'.format(param_key, estimator))
        return step_key, param_key, estimator

    def set_params(self, **params):
        for k, v in params.items():
            step_key, param_key, estimator = self._split_key(k)
            estimator.set_params(**{param_key: v})
        return self

    def fit(self, *args, **kwargs):
        return self._run_steps(*args, **dict(sklearn_method='fit', **kwargs))

    def transform(self, *args, **kwargs):
        return self._run_steps(*args, **dict(sklearn_method='transform', **kwargs))

    def fit_transform(self, *args, **kwargs):
        return self._run_steps(*args, **dict(sklearn_method='fit_transform', **kwargs))

    def ensemble_fit(self, X=None, y=None, sample_weight=None, ngen=3,
                     sampler=None, args_gen=None, client=None,
                     init_ensemble_size=1, ensemble_init_func=None,
                     models_share_sample=True,
                     model_selection=None, model_selection_kwargs=None,
                     scoring=None, scoring_kwargs=None, method='fit',
                     partial_fit_batches=1, classes=None, serialize_models=None,
                     **kwargs):
        return ensemble(self, ngen, X=X, y=y, sample_weight=sample_weight,
                     sampler=sampler, args_gen=args_gen, client=client,
                     init_ensemble_size=init_ensemble_size,
                     ensemble_init_func=ensemble_init_func,
                     models_share_sample=models_share_sample,
                     model_selection=model_selection,
                     model_selection_kwargs=model_selection_kwargs,
                     scoring=scoring, scoring_kwargs=scoring_kwargs,
                     method=method, partial_fit_batches=partial_fit_batches,
                     classes=classes, serialize_models=serialize_models,
                     **kwargs)

    def fit_transform_ensemble(self, *args, **kwargs):
        kw = dict(**kwargs)
        kw['method'] = 'fit_transform'
        return self.ensemble_fit(*args, **kw)

    def transform_ensemble(self, *args, **kwargs):
        kw = dict(**kwargs)
        kw['method'] = 'transform'
        return self.ensemble_fit(*args, **kw)

    def fit_ea(self, method, *args, **kwargs):
        raise NotImplementedError('')

    def predict(self, X=None, y=None, sample_weight=None, **kwargs):
        args = (X,)
        kw = dict(y=y, sample_weight=sample_weight,
                  sklearn_method='predict',**kwargs)
        return self._run_steps(*args, **kw)

    def fit_and_predict(self, *args, **kwargs):
        return self.fit(*args, **kwargs).predict(*args, **kwargs)

    @wraps(predict_many)
    def predict_many(self, *args, **kwargs):
        return predict_many(*args, **kwargs)
