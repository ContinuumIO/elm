
from collections import Sequence
from functools import partial, wraps
import copy
import logging

import numpy as np
import xarray as xr
from sklearn.externals import joblib
from sklearn.exceptions import NotFittedError

from elm.model_selection import get_args_kwargs_defaults
from elm.model_selection.scoring import score_one_model
from elm.readers import ElmStore
from elm.pipeline.predict_many import predict_many
from elm.pipeline import steps as STEPS
from elm.pipeline.ensemble import ensemble as _ensemble
from elm.pipeline.util import _next_name

logger = logging.getLogger(__name__)

class Pipeline(object):

    def __init__(self, steps, scoring=None, scoring_kwargs=None):
        self._re_init_args_kwargs = copy.deepcopy(((steps,), dict(scoring=scoring, scoring_kwargs=scoring_kwargs)))
        self.steps = steps
        self._validate_steps()
        self._names = [_[0] for _ in self.steps]
        self.scoring_kwargs = scoring_kwargs
        self.scoring = scoring

    def new_with_params(self, **new_params):
        new = self.unfitted_copy()
        new.set_params(**new_params)
        return new

    def unfitted_copy(self):
        return Pipeline(*self._re_init_args_kwargs[0],
                        **self._re_init_args_kwargs[1])

    def _get_pipe_params(self):
        return {'scoring': self.scoring,
                'scoring_kwargs': self.scoring_kwargs,
                'steps': self.steps,}

    def _set_pipe_params(self, **params):
        p2 = self._get_pipe_params()
        p2.update(**params)
        return p2

    def _set_new_estimators(self):
        new_steps = []
        for name, est in self.steps:
            new = est.__new__(est.__class__)
            init_params = est.get_params()
            print(name, 'init_params', init_params)
            new.__init__(**init_params)
            new_steps.append((name, new))
        self.steps[:] = new_steps
        self._estimator = new_steps[-1][1]


    def _run_steps(self, X=None, y=None,
                  sample_weight=None,
                  sampler=None, args_list=None,
                  sklearn_method='fit',
                  method_kwargs=None,
                  new_params=None,
                  partial_fit_batches=1,
                  return_X=False,
                  **data_source):

        from elm.sample_util.sample_pipeline import _split_pipeline_output
        method_kwargs = method_kwargs or {}
        if y is None:
            y = method_kwargs.get('y')
        if sample_weight is None:
            sample_weight = method_kwargs.get('y')
        if not 'predict' in sklearn_method:
            prepare_for = 'train'
        else:
            prepare_for = 'predict'
        if new_params:
            self = self.unfitted_copy(**new_params)
        fit_func = None
        if X is None and y is None and sample_weight is None:
            X, y, sample_weight = self.create_sample(X=X, y=y, sampler=sampler,
                                                     args_list=args_list,
                                                     **data_source)
        for idx, (_, step_cls) in enumerate(self.steps[:-1]):

            if prepare_for == 'train':
                fit_func = step_cls.fit_transform
            else:
                fit_func = step_cls.transform
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
                                                   y=y,
                                                   prepare_for=prepare_for,
                                                   sample_weight=sample_weight,
                                                   method_kwargs=method_kwargs)


        else:
            kwargs = {'y': y, 'sample_weight': sample_weight}
            args = (X,)
        if 'predict' in sklearn_method:
            X = args[0]
            pred = fitter_or_predict(X.flat.values, **kwargs)
            if return_X:
                return pred, X
            return pred

        output = fitter_or_predict(*args, **kwargs)
        if sklearn_method in ('fit', 'partial_fit'):
            self._score_estimator(X, y=y, sample_weight=sample_weight)
            return self
        # transform or fit_transform most likely
        return _split_pipeline_output(output, X, y, sample_weight, 'fit_transform')

    def _post_run_pipeline(self, fitter_or_predict, estimator,
                           X, y=None, sample_weight=None, prepare_for='train',
                           method_kwargs=None):
        from elm.sample_util.sample_pipeline import final_on_sample_step
        return final_on_sample_step(fitter_or_predict,
                         estimator, X,
                         method_kwargs or {},
                         y=y if y is not None else method_kwargs.get('y'),
                         prepare_for=prepare_for,
                         sample_weight=sample_weight,
                         require_flat=True,
                      )

    def create_sample(self, **data_source):
        from elm.sample_util.sample_pipeline import create_sample_from_data_source
        from elm.sample_util.sample_pipeline import _split_pipeline_output
        X = data_source.get("X", None)
        y = data_source.get('y', None)
        sample_weight = data_source.get('sample_weight', None)
        if not ('sampler' in data_source or 'args_list' in data_source):
            if not any(_ is not None for _ in (X, y, sample_weight)):
                raise ValueError('Expected "sampler" or "args_list" in "data_source" or X, y, and/or sample_weight')
        if data_source.get('sampler') and X is None and y is None:
            output = create_sample_from_data_source(**data_source)
        else:
            output = (X, y, sample_weight)
        out = _split_pipeline_output(output, X=X, y=y,
                           sample_weight=sample_weight,
                           context=getattr(self, '_context', repr(data_source)))
        return out

    def _validate_steps(self):
        validated = []
        for idx, s in enumerate(self.steps):
            if not isinstance(s, Sequence):
                name, estimator = 'step_{}'.format(idx), s
            elif len(s) == 2:
                name, estimator = s
            else:
                raise ValueError('Expected steps in Pipeline to be list of 2-tuples (name, estimator) or list of estimators')
            validated.append((name, estimator))
            if idx < len(self.steps) - 1:
                if not isinstance(estimator, STEPS.StepMixin):
                    raise ValueError('Each step before the final one in an elm.pipeline.Pipeline must be an instance of a class from elm.pipeline.steps.  Found {}'.format(estimator))
                if not hasattr(estimator, 'fit_transform') and not (hasattr(estimator, 'fit') and hasattr(estimator, 'transform')):
                    raise ValueError("{} has no attribute 'fit_transform', or 'fit' and 'transform'".format(estimator))
            else:
                if not any(hasattr(estimator, m) for m in ('fit', 'partial_fit', 'fit_transform')):
                    raise ValueError('')
        self.steps = validated
        self._estimator = self.steps[-1][-1]

    def get_params(self, **kwargs):
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

    def partial_fit(self, *args, **kwargs):
        return self._run_steps(*args, **dict(sklearn_method='partial_fit', **kwargs))

    def transform(self, *args, **kwargs):
        return self._run_steps(*args, **dict(sklearn_method='transform', **kwargs))

    def fit_transform(self, *args, **kwargs):
        return self._run_steps(*args, **dict(sklearn_method='fit_transform', **kwargs))

    def fit_ensemble(self, X=None, y=None, sample_weight=None, ngen=3,
                     sampler=None, args_list=None, client=None,
                     init_ensemble_size=1, ensemble_init_func=None,
                     saved_ensemble_size=None,
                     models_share_sample=True,
                     model_selection=None, model_selection_kwargs=None,
                     scoring=None, scoring_kwargs=None, method='fit',
                     partial_fit_batches=1,
                     serialize_pipe=None,
                     method_kwargs=None,
                     **data_source):
        data_source = dict(X=X, y=y, sample_weight=sample_weight, sampler=sampler,
                           args_list=args_list, **data_source)
        self.ensemble = _ensemble(self, ngen, client=client,
                         init_ensemble_size=init_ensemble_size,
                         saved_ensemble_size=saved_ensemble_size,
                         ensemble_init_func=ensemble_init_func,
                         models_share_sample=models_share_sample,
                         model_selection=model_selection,
                         model_selection_kwargs=model_selection_kwargs,
                         scoring=scoring, scoring_kwargs=scoring_kwargs,
                         method=method, partial_fit_batches=partial_fit_batches,
                         serialize_pipe=serialize_pipe,
                         method_kwargs=method_kwargs,
                         **data_source)
        return self

    def fit_transform_ensemble(self, *args, **kwargs):
        kw = dict(**kwargs)
        kw['method'] = 'fit_transform'
        return self.fit_ensemble(*args, **kw)

    def transform_ensemble(self, *args, **kwargs):
        kw = dict(**kwargs)
        kw['method'] = 'transform'
        return self.fit_ensemble(*args, **kw)

    def fit_ea(self, X=None, y=None, sample_weight=None, ngen=3,
               evo_params=None, sampler=None, args_list=None, client=None,
               init_ensemble_size=1, ensemble_init_func=None,
               saved_ensemble_size=None,
               models_share_sample=True,
               model_selection=None, model_selection_kwargs=None,
               scoring=None, scoring_kwargs=None, method='fit',
               partial_fit_batches=1,
               serialize_pipe=None,
               method_kwargs=None,
               **data_source):
        from elm.pipeline.evolve_train import evolve_train
        if evo_params is None:
            raise ValueError('Expected evo_params to be not None (an instance of EvoParams)')
        data_source = dict(X=X, y=y, sample_weight=sample_weight, sampler=sampler,
                           args_list=args_list, **data_source)
        models = evolve_train(self,
                 ngen,
                 evo_params,
                 client=client,
                 init_ensemble_size=init_ensemble_size,
                 saved_ensemble_size=saved_ensemble_size,
                 ensemble_init_func=ensemble_init_func,
                 models_share_sample=models_share_sample,
                 model_selection=model_selection,
                 model_selection_kwargs=model_selection_kwargs,
                 scoring=scoring,
                 scoring_kwargs=scoring_kwargs,
                 method=method,
                 partial_fit_batches=partial_fit_batches,
                 method_kwargs=method_kwargs,
                 **data_source)
        self.ensemble = models
        return self.ensemble

    def predict(self, *args, **kwargs):
        kw = dict(sklearn_method='predict',**kwargs)
        return self._run_steps(*args, **kw)

    def fit_and_predict(self, *args, **kwargs):
        return self.fit(*args, **kwargs).predict(*args, **kwargs)

    @wraps(predict_many)
    def predict_many(self, X=None, y=None, sampler=None, args_list=None,
                     client=None, ensemble=None, to_cube=True, tag=None,
                     elm_train_path=None, saved_model_tag=None,
                     serialize=None, **data_source):
        ensemble = ensemble or self.ensemble
        if not ensemble:
            raise ValueError('Must call fit_ensemble first or give ensemble=<a list of ("tag_0", fitted_estimator) tuples>')
        data_source = dict(X=X, y=y, sampler=sampler, args_list=args_list, **data_source)
        return predict_many(data_source, tagged_models=ensemble,
                 client=client,
                 serialize=serialize,
                 to_cube=to_cube,
                 saved_model_tag=saved_model_tag,
                 elm_train_path=elm_train_path)


    def _score_estimator(self, X, y=None, sample_weight=None):
        if not self.scoring:
            if self.scoring_kwargs:
                raise ValueError("scoring_kwargs ignored if scoring is not given")
            return
        kw = self.scoring_kwargs or {}
        kw['y'] = y
        kw['sample_weight'] = sample_weight
        fit_args = (X,)
        score_one_model(self, self.scoring, *fit_args, **kw)

    def __repr__(self):
        strs = ('{}: {}'.format(*s) for s in self.steps)
        return '<elm.pipeline.Pipeline> with steps:\n' + '\n'.join(strs)

    def save(self, filename):
        return joblib.dump(self, filename)

    @classmethod
    def load(self, filename):
        return joblib.load(filename)




    __str__ = __repr__

