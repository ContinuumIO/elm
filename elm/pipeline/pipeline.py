from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABCMeta, abstractmethod
from collections import defaultdict
import os

import numpy as np
from scipy import sparse

from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Parallel, delayed, Memory
from sklearn.externals import six
from sklearn.utils import tosequence
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils import Bunch

from sklearn.pipeline import (Pipeline as sk_Pipeline,
                              _fit_transform_one,
                              _transform_one,
                              _fit_one_transformer,)
from elm.mldataset.wrap_sklearn import (_as_numpy_arrs,
                                        _from_numpy_arrs,
                                        get_row_index,
                                        SklearnMixin)

from sklearn.utils.metaestimators import _BaseComposition
from xarray_filters.pipeline import Step
from xarray_filters.func_signatures import filter_args_kwargs

__all__ = ['Pipeline', 'FeatureUnion']


class Pipeline(sk_Pipeline):

    # Estimator interface


    def _sk_method(self, method):
        return getattr(super(Pipeline, self), method)

    _as_numpy_arrs = _as_numpy_arrs
    _from_numpy_arrs = _from_numpy_arrs

    def _astype(self, step, X, y=None):
        astype = 'numpy'
        if not isinstance(step, Step):
            print('Numpy')
            X, y, row_idx = self._as_numpy_arrs(X, y)
            if row_idx is not None:
                self.row_idx = row_idx
        return X, y

    #def _validate_steps(self):
     #   return True

    def _do_this_step(self, step_idx):
        name, est = self.steps[step_idx]
        self._generic = {}
        for name, est in self.steps:
            if isinstance(est, Step):
                self._generic[name] = True
            else:
                self._generic[name] = False
        print('GEn', self._generic, name)
        do_step = True
        if getattr(self, '_run_generic_only', None) is None:
            pass
        else:
            if self._run_generic_only and not name in self._generic:
                do_step = False
        if getattr(self, '_skip_generic', None) is None:
            pass
        else:
            if self._skip_generic and name in self._generic:
                do_step = False
        print('do_step', name, do_step)
        return do_step

    def _fit_generic_only(self, X, y, **fit_params):
        self._generic = {}
        for name, est in self.steps:
            if isinstance(est, Step):
                self._generic[name] = True
            else:
                self._generic[name] = False


    def _fit(self, X, y=None, **fit_params):

        self._validate_steps()
        # Setup the memory
        memory = self.memory
        if memory is None:
            memory = Memory(cachedir=None, verbose=0)
        elif isinstance(memory, six.string_types):
            memory = Memory(cachedir=memory, verbose=0)
        elif not isinstance(memory, Memory):
            raise ValueError("'memory' should either be a string or"
                             " a joblib.Memory instance, got"
                             " 'memory={!r}' instead.".format(memory))

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for step_idx, (name, transformer) in enumerate(self.steps[:-1]):
            #if self._do_this_step(step_idx):
            Xt, y = self._astype(transformer, Xt, y=y)
            print('Types', step_idx, [type(_) for _ in (Xt, y)])
            if transformer is None:
                pass
            else:
                if memory.cachedir is None:
                    # we do not clone when caching is disabled to preserve
                    # backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
                # Fit or load from cache the current transfomer
                Xt, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer, None, Xt, y,
                    **fit_params_steps[name])
                # Replace the transformer of the step with the fitted
                # transformer. This is necessary when loading the transformer
                # from the cache.
                self.steps[step_idx] = (name, fitted_transformer)
        if self._final_estimator is None:
            return Xt, {}
        fit_params = fit_params_steps[self.steps[-1][0]]
        return Xt, y, fit_params

    def fit(self, X, y=None, **fit_params):
        """Fit the model

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : Pipeline
            This estimator
        """

        Xt, y, fit_params = self._fit(X, y, **fit_params)
        if self._final_estimator is not None:
            Xt, y = self._astype(self._final_estimator, Xt, y=y)
            self._final_estimator.fit(Xt, y, **fit_params)
        return self

    def _as_dataset(self, as_dataset, y, row_idx, features_layer=None):

        if as_dataset:
            return self._from_numpy_arrs(y, row_idx, features_layer=features_layer)
        return y

    def _before_predict(self, method, X, y=None, **fit_params):

        Xt = X
        for step_idx, (name, transform) in enumerate(self.steps[:-1]):
            if transform is not None:
                #if not self._do_this_step(step_idx):
                 #   continue
                Xt, y = self._astype(transform, Xt, y=y)
                Xt = transform.transform(Xt)
        final_estimator = self.steps[-1][-1]
        fit_params = dict(row_idx=self.row_idx, **fit_params)
        if y is not None:
            fit_params['y'] = y
        if hasattr(self, 'row_idx'):
            fit_params['row_idx'] = self.row_idx
        fit_params = filter_args_kwargs(getattr(self._final_estimator, method),
                                        Xt,
                                        **fit_params)
        return Xt, y, fit_params, final_estimator

    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X, as_dataset=True):
        """Apply transforms to the data, and predict with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_pred : array-like
        """
        Xt, _, fit_params, final_estimator = self._before_predict('predict',
                                                                  X, y=None)
        y = final_estimator.predict(**fit_params)
        return self._as_dataset(as_dataset, y, self.row_idx, features_layer='predict')

    @if_delegate_has_method(delegate='_final_estimator')
    def fit_predict(self, X, y=None, as_dataset=True, **fit_params):
        """Applies fit_predict of last step in pipeline after transforms.

        Applies fit_transforms of a pipeline to the data, followed by the
        fit_predict method of the final estimator in the pipeline. Valid
        only if the final estimator implements fit_predict.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : array-like
        """
        Xt, y, fit_params, final_estimator = self._before_predict('fit_predict',
                                                                  X, y=y,
                                                                  **fit_params)
        y = final_estimator.fit_predict(**fit_params)
        return self._as_dataset(as_dataset, y, self.row_idx, features_layer='predict')

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_proba(self, X):
        """Apply transforms, and predict_proba of the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]
        """
        Xt, _, fit_params, final_estimator = self._before_predict('predict_proba',
                                                                  X, y=None,
                                                                  **fit_params)
        prob_a = final_estimator.predict_proba(**fit_params)
        return self._as_dataset(as_dataset, prob_a, self.row_idx, features_layer='proba')

    @if_delegate_has_method(delegate='_final_estimator')
    def decision_function(self, X):
        """Apply transforms, and decision_function of the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """
        Xt, _, fit_params, final_estimator = self._before_predict('decision_function',
                                                                  X, y=None,
                                                                  **fit_params)
        d = final_estimator.decision_function(**fit_params)
        return self._as_dataset(as_dataset, d, self.row_idx, features_layer='decision')

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_log_proba(self, X):
        """Apply transforms, and predict_log_proba of the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """
        Xt, y, fit_params, final_estimator = self._before_predict('predict_log_proba',
                                                                  X, y=y,
                                                                  **fit_params)
        log_proba = final_estimator.predict_log_proba(**fit_params)
        return self._as_dataset(as_dataset, log_proba, self.row_idx, features_layer='log_proba')

    @if_delegate_has_method(delegate='_final_estimator')
    def score(self, X, y=None, sample_weight=None):
        """Apply transforms, and score with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.

        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.

        Returns
        -------
        score : float
        """
        Xt, y, fit_params, final_estimator = self._before_predict('score',
                                                                  X, y=y,
                                                                  **fit_params)
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        return final_estimator.score(Xt, y, **score_params)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator

        Fits all the transforms one after the other and transforms the
        data, then uses fit_transform on transformed data with the final
        estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
            Transformed samples
        """
        last_step = self._final_estimator
        Xt, y, fit_params = self._fit(X, y, **fit_params)
        if hasattr(last_step, '_cls'):
            has_ft = hasattr(last_step._cls, 'fit_transform')
        else:
            has_ft = hasattr(last_step, 'fit_transform')
        #skip = getattr(self, '_run_generic_only', False)
        #if skip:
        #    return X, y
        if last_step is None:
            return Xt
        elif has_ft:
            return last_step.fit_transform(Xt, y, **fit_params)
        else:
            return last_step.fit(Xt, y, **fit_params).transform(Xt)

