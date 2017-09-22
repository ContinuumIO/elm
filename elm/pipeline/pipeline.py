"""
The :mod:`sklearn.pipeline` module implements utilities to build a composite
estimator, as a chain of transforms and estimators.
"""
# Author: Edouard Duchesnay
#         Gael Varoquaux
#         Virgile Fritsch
#         Alexandre Gramfort
#         Lars Buitinck
# License: BSD

from collections import defaultdict

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import sparse

from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Parallel, delayed, Memory
from sklearn.externals import six
from sklearn.utils import tosequence
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils import Bunch

from sklearn.pipeline import Pipeline as sk_Pipeline
from elm.model_selection.sklearn_mldataset import (_call_sk_method,
                                                   _as_numpy_arrs,
                                                   _from_numpy_arrs,
                                                   get_row_index)

from sklearn.utils.metaestimators import _BaseComposition

__all__ = ['Pipeline', 'FeatureUnion']


class Pipeline(sk_Pipeline):

    # Estimator interface

    def _sk_method(self, method):
        return getattr(super(Pipeline, self), method)

    _as_numpy_arrs = _as_numpy_arrs
    _from_numpy_arrs = _from_numpy_arrs

    def _fit(self, X, y=None, **fit_params):
        X, y, self.row_idx = self._as_numpy_arrs(X, y=y)
        return self._sk_method('_fit')(X, y=y, **fit_params)

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

        Xt, fit_params = self._fit(X, y, **fit_params)
        return self._sk_method('_fit')(Xt)


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
        Xt, _, row_idx = self._as_numpy_arrs(X)
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)
                Xt, _, row_idx = self._as_numpy_arrs(Xt)
        y = self.steps[-1][-1].predict(Xt)

    def _as_dataset(self, as_dataset, X, y, row_idx, features_layer=None):

        if as_dataset:
            _, y, _ = _from_numpy_arrs(X, y, row_idx, features_layer=features_layer)
        return y

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
        Xt, fit_params = self._fit(X, y, **fit_params)
        row_idx = getattr(self, 'row_idx', None)
        if row_idx is None:
            self.row_idx = get_row_index(X)
        y = self.steps[-1][-1].fit_predict(Xt, y, **fit_params)
        return self._as_dataset(as_dataset, X, y, self.row_idx)

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
        Xt, _, _ = self._as_numpy_arrs(X)
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)
                Xt, _, _ = self._as_numpy_arrs(Xt)
        prob_a = self.steps[-1][-1].predict_proba(Xt)
        return self._as_dataset(as_dataset, X, prob_a, self.row_idx)

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
        Xt, _, _ = self._as_numpy_arrs(X)
        d = self._sk_method('decision_function')(Xt)
        return self._as_dataset(as_dataset, X, d, self.row_idx)

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
        Xt, _, row_idx = self._as_numpy_arrs(X)
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)
                Xt, _, row_idx = self._as_numpy_arrs(Xt)
        log_proba = self.steps[-1][-1].predict_log_proba(Xt)
        row_idx = getattr(self, 'row_idx', None)
        return self._as_dataset(as_dataset, X, log_proba, row_idx)

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
        Xt, _, row_idx = self._as_numpy_arrs(X)
        return self._sk_method('score')(Xt)



