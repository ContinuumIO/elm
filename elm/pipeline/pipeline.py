from __future__ import absolute_import, division, print_function, unicode_literals

'''
----------------------

``elm.pipeline.Pipeline``
~~~~~~~~~~~~~~~~~~~~~~~~~

Run a series of transformations on a series of samples, using
ensemble approach or evolutionary algorithms (EA) that have model
scoring selection logic.

Ensemble or EA methods may be combined with partial_fit methods,
including partial_fit of transformations before partial_fit of a final
estimator.  See the example below with IncrementalPCA before
MiniBatchKMeans.

Dask graphs are used for parallelism - pass the "client" argument
to methods such

Each sample in the series of samples is expressed as a tuple::

    (X, y, sample_weight)

with X as an earthio.ElmStore
and y and sample_weight as a numpy arrays or None if not needed.

``elm.pipeline.Pipeline`` is similar to scikit-learn's Pipeline concept
(sklearn.pipeline.Pipeline) in usage

'''
from collections import Sequence
from functools import partial
import copy
import logging

import dill
try:
    from earthio import check_X_data_type
except:
    check_X_data_type = None # TODO handle case where earthio not installed
import numpy as np
import xarray as xr
from sklearn.exceptions import NotFittedError

from elm.model_selection import get_args_kwargs_defaults
from elm.model_selection.scoring import score_one_model
from elm.pipeline.predict_many import predict_many
from elm.pipeline import steps as STEPS
from elm.pipeline.ensemble import ensemble as _ensemble
from elm.pipeline.util import _next_name


logger = logging.getLogger(__name__)

class Pipeline(object):
    '''
    Pipeline of transformation, fit steps for
    ensemble, evolutionary and/or partial_fit with dask

    Parameters:
        :steps:  Steps or transformers and final estimators. Formats:

                - Each step may be a tuple with a label and estimator/transformer \('kmeans', KMeans\(n_clusters\)\) or just an estimator
                - All transformers except the last, must be from: ``elm.pipeline.steps``, e.g. ``elm.pipeline.steps.Transform(PCA(), partial_fit_batches=2)``
                - The final estimator is typically from sklearn, with fit and predict methods

        :scoring: Scoring function. See model scoring functions in scikit-learn docs
                  http://scikit-learn.org/stable/modules/model_evaluation.html.
                  Also see a custom scoring example in :any:``elm.model_selection.kmeans.kmeans_aic``
        :scoring_kwargs: Keyword args passed to scoring
    '''


    def __init__(self, steps, scoring=None, scoring_kwargs=None):
        '''
        Pipeline of transformation, fit steps for
        ensemble, evolutionary and/or partial_fit with dask

        Parameters:
            :steps:  Steps or transformers and final estimators. Formats:

                    - Each step may be a tuple with a label and estimator/transformer
                    ('kmeans', KMeans(n_clusters)) or just an estimator
                    - All transformers except the last, must be from:

                        - elm.pipeline.steps, e.g.
                        - elm.pipeline.steps.Transform(PCA(), partial_fit_batches=2)

                    - The final estimator is typically from sklearn, with
                    fit and predict methods

            :scoring: Scoring function. See model scoring functions
                     in scikit-learn docs:
                        http://scikit-learn.org/stable/modules/model_evaluation.html

                     And also see an custom scoring example in:
                        :func:``elm.model_selection.kmeans.kmeans_aic``

            :scoring_kwargs: Keyword args passed to scoring
        '''
        self._re_init_args_kwargs = copy.deepcopy(((steps,), dict(scoring=scoring, scoring_kwargs=scoring_kwargs)))
        self.steps = steps
        self._validate_steps()
        self._names = [_[0] for _ in self.steps]
        self.scoring_kwargs = scoring_kwargs
        self.scoring = scoring

    def new_with_params(self, **new_params):
        '''Return a copy of this Pipeline as it was initialized,
        but with new_params'''
        new = self.unfitted_copy()
        new.set_params(**new_params)
        return new

    def unfitted_copy(self):
        '''Return copy of original Pipeline that was created (self
           before modifications)'''
        return Pipeline(*self._re_init_args_kwargs[0],
                        **self._re_init_args_kwargs[1])

    def _run_steps(self, X=None, y=None,
                  sample_weight=None,
                  sampler=None, args_list=None,
                  sklearn_method='fit',
                  method_kwargs=None,
                  new_params=None,
                  partial_fit_batches=1,
                  return_X=False,
                  **data_source):
        '''Evaluate each fit/transform step in self.steps.  Used
        by fit, transform, predict and related methods'''
        from elm.sample_util.sample_pipeline import _split_pipeline_output
        method_kwargs = method_kwargs or {}
        if y is None:
            y = method_kwargs.get('y')
        if sample_weight is None:
            sample_weight = method_kwargs.get('sample_weight')
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
        else:
            X, y, sample_weight = _split_pipeline_output(X, X, y, sample_weight, sklearn_method)
        for idx, (_, step_cls) in enumerate(self.steps[:-1]):

            if prepare_for == 'train':
                fit_func = step_cls.fit_transform
            else:
                fit_func = step_cls.transform
                if not hasattr(getattr(step_cls, '_estimator', None), 'transform'):
                    # Estimator such as TSNE with no transform method, just fit_transform
                    fit_func = step_cls.fit_transform
            func_out = fit_func(X, y=y, sample_weight=sample_weight)
            if func_out is not None:
                X, y, sample_weight = _split_pipeline_output(func_out, X, y,
                                                       sample_weight, repr(fit_func))

        if getattr(sample_weight, 'ndim', None) == 2:
            sample_weight = sample_weight[:, 0]
        check_X_data_type(X)
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
            # Drop sample_weight kwarg, since it's only needed for fitting (not predicting)
            kwargs.pop('sample_weight', None)
            pred = fitter_or_predict(X.flat.values, **kwargs)
            if return_X:
                return pred, X
            return pred
        output = fitter_or_predict(*args, **kwargs)
        if sklearn_method in ('fit', 'partial_fit', 'fit_predict'):
            kw = kwargs.copy()
            kw.update((self.scoring_kwargs or {}).copy())
            self._score_estimator(*args, **kw)
            return self
        # transform or fit_transform most likely
        return _split_pipeline_output(output, X, y, sample_weight, 'fit_transform')

    def _post_run_pipeline(self, fitter_or_predict, estimator,
                           X, y=None, sample_weight=None, prepare_for='train',
                           method_kwargs=None):
        '''Finally convert X ElmStore to X numpy array for
        sklearn estimator in last step'''
        from elm.sample_util.sample_pipeline import final_on_sample_step
        method_kwargs = method_kwargs or {}
        y = y if y is not None else method_kwargs.get('y')
        return final_on_sample_step(fitter_or_predict,
                         estimator, X,
                         method_kwargs,
                         y=y,
                         prepare_for=prepare_for,
                         sample_weight=sample_weight,
                         require_flat=True,
                      )

    def create_sample(self, **data_source):
        '''
        Standardizes the output of a sampler or X, y, sample_weight
        to be a tuple of (X, y, sample_weight)

        Calls the sampler if given in data_source.  Typically the sampler
        takes

        :\*\*data_source: should have:

            :sampler: a function taking (\*args, \*\*kwargs), returning an
                      X ElmStore or tuple of (X, y, sample_weight) with X as
                      ElmStore.  Arguments to the sampler are sampler_args
                      and \*\*data_source is also passed.
            :sampler_args: if passed to this function, sampler_args are typically
                         created by unpacking of each of element of "args_list" given
                         to other methods in this class.

        OR the \*\*data_source may have:

            - :X:, :y:, and/or :sample_weight: keys/values, with X as an ElmStore,
                in which case, this function just passes them through.  See
                usage in ensemble
        '''
        from elm.sample_util.sample_pipeline import create_sample_from_data_source
        from elm.sample_util.sample_pipeline import _split_pipeline_output
        X = data_source.get("X", None)
        y = data_source.get('y', None)
        logger.info('Call create_sample')
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
        '''Give every step a name if not given, raise ValueError if needed'''
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
        '''Return the params dict for each estimator/transformer in pipeline'''
        params = {}
        for name, estimator in self.steps:
            for k, v in estimator.get_params().items():
                params['{}__{}'.format(name, k)] = v
        return params

    def _validate_key(self, k, context=''):
        '''Make sure a key/name is in the named pipeline steps'''
        if not k in self._names:
            raise ValueError('{} parameter - {} is not in {}'.format(context, k, self._names))

    def _split_key(self, k):
        '''Split a param key such as kmeans__n_clusters. Raise ValueError if needed'''
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
        '''
        Set the parameters of the Pipeline.  See also new_with_params
        to ensure a new unfitted_copy of Pipeline is returned with
        new initialization parameters.
        '''
        new_steps = {}
        for k, v in params.items():
            step_key, param_key, estimator = self._split_key(k)
            estimator.set_params(**{param_key: v})
            new_steps[step_key] =  estimator
        steps = []
        for tag, s in self.steps:
            if tag in new_steps:
                steps.append((tag, new_steps[tag]))
            else:
                steps.append((tag, s))
        return Pipeline(steps, **self._re_init_args_kwargs[1])

    def fit(self, *args, **kwargs):
        return self._run_steps(*args, **dict(sklearn_method='fit', **kwargs))

    def partial_fit(self, *args, **kwargs):
        return self._run_steps(*args, **dict(sklearn_method='partial_fit', **kwargs))

    def transform(self, *args, **kwargs):
        return self._run_steps(*args, **dict(sklearn_method='transform', **kwargs))

    def fit_transform(self, *args, **kwargs):
        return self._run_steps(*args, **dict(sklearn_method='fit_transform', **kwargs))

    def fit_predict(self, *args, **kwargs):
        return self._run_steps(*args, **dict(sklearn_method='fit_predict', **kwargs))

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
        '''Run ensemble approach to fitting

        This function passes the Pipeline to :mod:``elm.pipeline.ensemble``

        See the argument spec there:
        ''' + _ensemble.__doc__
        data_source = dict(X=X, y=y, sample_weight=sample_weight, sampler=sampler,
                           args_list=args_list, **data_source)
        if scoring:
            self.scoring = scoring
        if scoring_kwargs:
            self.scoring_kwargs = scoring_kwargs
        self.ensemble = _ensemble(self, ngen, client=client,
                         init_ensemble_size=init_ensemble_size,
                         saved_ensemble_size=saved_ensemble_size,
                         ensemble_init_func=ensemble_init_func,
                         models_share_sample=models_share_sample,
                         model_selection=model_selection,
                         model_selection_kwargs=model_selection_kwargs,
                         scoring=self.scoring,
                         scoring_kwargs=self.scoring_kwargs,
                         method=method, partial_fit_batches=partial_fit_batches,
                         serialize_pipe=serialize_pipe,
                         method_kwargs=method_kwargs,
                         **data_source)
        return self

    def fit_ea(self, X=None, y=None, sample_weight=None,
               evo_params=None, sampler=None, args_list=None, client=None,
               init_ensemble_size=1, ensemble_init_func=None,
               saved_ensemble_size=None,
               models_share_sample=True,
               scoring=None, scoring_kwargs=None, method='fit',
               partial_fit_batches=1,
               serialize_pipe=None,
               method_kwargs=None,
               **data_source):

        '''Passes the Pipeline to :any:``elm.pipeline.evolve_train``

        Arguments for evolve_train and ensemble are similar,
        with the differences being:

        1) evolve_train requires evo_params argument, typically
            from ea_setup::

                from elm.model_selection import ea_setup
                param_grid =  {
                    'kmeans__n_clusters': list(range(3, 10)),
                    'top_n__percentile': list(range(20, 100, 5)),
                    'control': {
                        # methods from deap for selection, crossover, mutation
                        'select_method': 'selNSGA2',
                        'crossover_method': 'cxTwoPoint',
                        'mutate_method': 'mutUniformInt',
                        'init_pop': 'random', # 'random' is the only init_pop

                        # arguments passed to select, crossover and mutate
                        'indpb': 0.5, # if using cxUniform or cxUniformPartialyMatched
                        'mutpb': 0.9, # mutation prob
                        'cxpb':  0.3, # crossover prob
                        'eta':   20,  # passed to selNSGA2
                        'ngen':  2,   # number of generations
                        'mu':    4,   # population size (number of Pipeline instances)
                        'k':     4,   # select top k (NSGA-2)
                        # Control stopping on absolute change in objectives
                        # (agg controls application of abs change check to each
                           objective, is using multi-objective scoring)
                        'early_stop': {'abs_change': [10], 'agg': 'all'},
                        # Or on percent change or threshold in objective scores
                        # alternatively early_stop: {percent_change: [10], agg: all}
                        # alternatively early_stop: {threshold: [10], agg: any}
                    }
                }

            :evo_params: ea_setup(param_grid=param_grid, param_grid_name='param_grid_example',
                         score_weights=[-1]) # minimization

        2) ensemble takes "model_selection" and "model_selection_kwargs"
        while evolve_train takes evo_params, which typically uses selNSGA2
        from deap as a model selector.

        '''
        from elm.pipeline.evolve_train import evolve_train
        if evo_params is None:
            raise ValueError('Expected evo_params to be not None (an instance of EvoParams)')
        if saved_ensemble_size is None:
            saved_ensemble_size = evo_params.deap_params['control']['mu']
        data_source = dict(X=X, y=y, sample_weight=sample_weight, sampler=sampler,
                           args_list=args_list, **data_source)
        ngen = evo_params.deap_params['control']['ngen']

        if scoring:
            self.scoring = scoring
        if scoring_kwargs:
            self.scoring_kwargs = scoring_kwargs
        models = evolve_train(self,
                             evo_params,
                             client=client,
                             init_ensemble_size=init_ensemble_size,
                             saved_ensemble_size=saved_ensemble_size,
                             ensemble_init_func=ensemble_init_func,
                             models_share_sample=models_share_sample,
                             scoring=self.scoring,
                             scoring_kwargs=self.scoring_kwargs,
                             method=method,
                             partial_fit_batches=partial_fit_batches,
                             method_kwargs=method_kwargs,
                             **data_source)
        self.ensemble = models
        return self

    def predict(self, X=None, method_kwargs=None, return_X=False, **data_source):
        '''Call the final estimator's predict method

        This does not predict from all fitted ensemble members
        but rather one Pipeline instance.  To predict from
        all or part of an ensemble after using "fit_ensemble"
        or "fit_ea", use Pipeline.predict_many,
        not this Pipeline.predict.

        Parameters:
            :X: ElmStore or None if "data_source" in kwargs has
               a sampler and sampler_args keys/values
            :method_kwargs: kwargs to predict if any
            :return_X: also return the final X ElmStore ( the
                      X ElmStore with a Dataset "flat" whose
                      values are used in prediction)
            :data_source: if X is None, data_source must have a
               sampler_func and sampler_args
        Returns:
            :Numpy array:, typically 1-D, y predicted from final
            estimator in self.steps

        '''
        kw = dict(sklearn_method='predict', method_kwargs=method_kwargs,
                  return_X=return_X, **data_source)
        return self._run_steps(X, **kw)

    def fit_and_predict(self, *args, **kwargs):
        '''Calls fit and predict with same data_source.

        See also

         - :any:``Pipeline.fit``
         - :any:``Pipeline.predict``
        '''
        return self.fit(*args, **kwargs).predict(*args, **kwargs)

    def predict_many(self, X=None, sampler=None, args_list=None,
                     client=None, ensemble=None, to_raster=True,
                     saved_model_tag=None,
                     serialize=None, **data_source):
        '''
        Predict from an ensemble of models for fixed X or series of
        sampler calls.

        Parameters:
            :X:  If predicting one or more models on one X ElmStore sample,then pass X
            :sampler: If not passing X, pass sampler, a function called
                 on an unpacking each element of args_list.
            :args_list: If not passing X, each element of args_list is
               unpacked as \*args to sampler
            :client: dask distributed or ThreadPool client
            :ensemble: list of ``('tag_0', Pipeline)`` instances or ``None``
               to use the last "fit_ensemble" or "fit_ea" output
               trained models
            :to_raster: True means to convert the 1-D predicted Y to
               2-D Dataset (common use case is converting prediction
               to image view of classifier output in space)

               See also ``earthio.inverse_flatten`` which converts
               1-D y to 2-D Dataset and ElmStore.  inverse_flatten is
               called if to_raster is True
            :saved_model_tag: This is a tag for an ensemble. An ensemble
               is a list of ``(tag, Pipeline)`` tuples and ``saved_model_tag`` is
               the higher level tag.  This argument is used in the
               config file elm interface.  See also:

                :func:``elm.pipeline.parse_run_config``

            :serialize: None to return all prediction arrays (not
               feasible in many cases) or a serializer of arrays
               with this signature::

                   serialize(y=y, X=X_final, tag=predict_tag,
                             elm_predict_path=elm_predict_path,)

                where
                    :y: is an ElmStore either 1-D or 2-D (see to_raster)
                    :X_final: is the X ElmStore that was fit (the Pipeline
                        will preserve attrs in X useful for serializing y)
                    :tag: is a unique tag of sample, Pipeline instance and saved_model_tag
                    :elm_predict_path: is the root dir for serialization
                        output, defaulting to ELM_PREDICT_PATH from environment
                        variables
            :\*\*data_source: keyword args passed to the sampler on each call

        Returns:
            :preds: Sequence of predictions if serialize is None else Sequence
                   of the outputs from serialize

        '''
        ensemble = ensemble or self.ensemble
        if not ensemble:
            raise ValueError('Must call fit_ensemble first or give ensemble=<a list of ("tag_0", fitted_estimator) tuples>')
        data_source = dict(X=X, sampler=sampler, args_list=args_list, **data_source)
        return predict_many(data_source,
                 ensemble=ensemble,
                 client=client,
                 serialize=serialize,
                 to_raster=to_raster,
                 saved_model_tag=saved_model_tag)


    def _score_estimator(self, X, y=None, sample_weight=None, **kw):
        '''Run the scoring function with scoring_kwargs that were given in __init__
        '''
        if not self.scoring:
            if self.scoring_kwargs:
                raise ValueError("scoring_kwargs ignored if scoring is not given")
            return
        kw = kw.copy()
        kw['y'] = y
        kw['sample_weight'] = sample_weight
        fit_args = (X,)
        score_one_model(self, self.scoring, *fit_args, **kw)

    def __repr__(self):
        '''repr of each step'''
        strs = ('{}: {}'.format(*s) for s in self.steps)
        return '<elm.pipeline.Pipeline> with steps:\n' + '\n'.join(strs)

    def save(self, filename):
        '''save the Pipeline to filename

        Parameters:
            :filename: string filename

        Returns:
            None

        Uses dill.dump
        '''
        with open(filename, 'wb') as f:
            return dill.dump(self, f)

    @classmethod
    def load(self, filename):
        '''load a Pipeline from dill dump

        Parameters:
            :filename: string filename

        Returns:
            :Pipeline: fitted pipeline with "ensemble" attribute if fit_ensemble or fit_ea were called.
        '''
        with open(filename, 'rb') as f:
            return dill.load(f)

    __str__ = __repr__
