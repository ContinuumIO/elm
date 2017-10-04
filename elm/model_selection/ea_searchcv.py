from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import copy

import dask.array as da
from dask_searchcv.model_selection import (_DOC_TEMPLATE,
                                           RandomizedSearchCV,
                                           DaskBaseSearchCV)
import numpy as np
from elm.model_selection.evolve import (fit_ea,
                                        DEFAULT_CONTROL,
                                        ind_to_new_params,
                                        DEFAULT_EVO_PARAMS,)
from elm.mldataset.serialize_mixin import SerializeEstimator
from elm.mldataset.wrap_sklearn import SklearnMixin
from elm.model_selection.sorting import pareto_front
from elm.model_selection.base import base_selection
from elm.pipeline import Pipeline
from xarray_filters.func_signatures import filter_kw_and_run_init
from xarray_filters.constants import DASK_CHUNK_N
from xarray_filters import MLDataset
import xarray as xr


def _concat_cv_results(cv1, cv2, gen=0):
    cv_results = {}
    for k in cv2:
        if k in cv1:
            v1 = cv1[k]
        else:
            v1 = None
        v2 = cv2[k]
        if v1 is None:
            v = v2
        elif isinstance(v1, list):
            v = v1 + v2
        elif isinstance(v1, np.ndarray):
            v = np.concatenate((v1, v2)).squeeze()
        else:
            raise NotImplementedError('{} not handled ({})'.format(k, v1))
        cv_results[k] = v
        lenn = len(v2)
    gen_arr = cv1.get('gen', None)
    this_gen_arr = np.ones((lenn,)) * gen
    if gen_arr is None:
        cv_results['gen'] = this_gen_arr
    else:
        cv_results['gen'] = np.concatenate((gen_arr, this_gen_arr))

    return cv_results

_ea_oneliner = """\
Exhaustive search over specified parameter values for an estimator.\
"""
_ea_description = """\
The parameters of the estimator used to apply these methods are optimized
by cross-validated evolutionary algorithm search over a parameter grid.\
"""
_ea_parameters = """\
TODO : DESCRIBE OTHER PARAMETERS FOR EaSearchCV
"""
_ea_example = """\
>>> from sklearn import svm, datasets
>>> from elm.model_selection.ea_searchcv import EaSearchCV
>>> iris = datasets.load_iris()
>>> parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10]}
>>> svc = svm.SVC()
>>> clf = EaSearchCV(svc, parameters)
>>> clf.fit(iris.data, iris.target)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
EaSearchCV(cache_cv=..., cv=..., error_score=...,
        estimator=SVC(C=..., cache_size=..., class_weight=..., coef0=...,
                      decision_function_shape=..., degree=..., gamma=...,
                      kernel=..., max_iter=-1, probability=False,
                      random_state=..., shrinking=..., tol=...,
                      verbose=...),
        iid=..., n_jobs=..., param_distributions=..., refit=..., return_train_score=...,
        scheduler=..., scoring=...)
>>> sorted(clf.cv_results_.keys())  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
['mean_fit_time', 'mean_score_time', 'mean_test_score',...
 'mean_train_score', 'param_C', 'param_kernel', 'params',...
 'rank_test_score', 'split0_test_score',...
 'split0_train_score', 'split1_test_score', 'split1_train_score',...
 'split2_test_score', 'split2_train_score',...
 'std_fit_time', 'std_score_time', 'std_test_score', 'std_train_score'...]\
"""

class EaSearchCV(RandomizedSearchCV, SklearnMixin, SerializeEstimator):

    __doc__ = _DOC_TEMPLATE.format(name="EaSearchCV",
                                   oneliner=_ea_oneliner,
                                   description=_ea_description,
                                   parameters=_ea_parameters,
                                   example=_ea_example)

    def __init__(self, estimator, param_distributions, n_iter=10, ngen=3,
                 random_state=None,
                 scoring=None,
                 score_weights=None, model_selection=None,
                 sort_fitness=pareto_front,
                 model_selection_kwargs=None,
                 select_with_test=True,
                 avoid_repeated_params=True,
                 iid=True, refit=True,
                 cv=None, error_score='raise', return_train_score=True,
                 scheduler=None, n_jobs=-1, cache_cv=True):
        filter_kw_and_run_init(RandomizedSearchCV.__init__, **locals())
        self.ngen = ngen
        self.select_with_test = select_with_test
        self.model_selection = model_selection
        self.score_weights = score_weights
        self.avoid_repeated_params = avoid_repeated_params
        self.cv_results_all_gen_ = {}

    def _close(self):
        self.cv_results_ = getattr(self, 'cv_results_all_gen_', self.cv_results_)
        to_del = ('_ea_gen', 'cv_results_all_gen_',
                  '_invalid_ind', '_pop', '_evo_params',
                  '_toolbox')
        for attr in to_del:
            if hasattr(self, attr):
                delattr(self, attr)
        to_del_est = ('_skip_generic', '_run_generic_only')
        for attr in to_del:
            if hasattr(self.estimator, attr):
                delattr(self.estimator, attr)

    @property
    def _is_ea(self):
        model_selection = self.get_params()['model_selection']
        if not model_selection or isinstance(model_selection, dict):
            return True
        return False

    @property
    def _model_selection(self):
        params = self.get_params()
        model_selection = params['model_selection']
        if not model_selection:
            model_selection = {}
        if isinstance(model_selection, dict):
            model_selection = model_selection.copy()
            for k, v in DEFAULT_EVO_PARAMS.items():
                if k not in model_selection:
                    model_selection[k] = v
            return model_selection
        kw = params['model_selection_kwargs'] or {}
        sort_fitness = params['sort_fitness'] or pareto_front
        score_weights = params.get('score_weights', (1,))
        selector = partial(base_selection,
                           model_selection=model_selection,
                           sort_fitness=sort_fitness,
                           score_weights=score_weights,
                           **kw)
        return selector

    def _within_gen_param_iter(self, gen=0):
        if not self._is_ea:
            for params in getattr(self, 'next_params_', []):
                yield params
            return
        deap_params = self._evo_params['deap_params']
        if gen == 0:
            invalid_ind = self._pop
        else:
            invalid_ind = self._invalid_ind
        for idx, ind in enumerate(invalid_ind):
            yield ind_to_new_params(deap_params, ind)

    def _fitnesses_to_deap(self, fitnesses):
        if isinstance(fitnesses, np.ndarray) and fitnesses.squeeze().ndim == 1:
            fitnesses = [(x,) for x in fitnesses]
        else:
            raise NotImplementedError('Multi-objective optimization would require a few changes to cv_results?')
        return fitnesses

    def _get_cv_scores(self):
        cv_results = getattr(self, 'cv_results_', None)
        if not cv_results:
            raise ValueError('Expected "cv_results_" attribute')
        if self.select_with_test:
            score_field = 'mean_test_score'
        else:
            score_field = 'mean_train_score'
        return self._fitnesses_to_deap(cv_results[score_field])

    def _open(self):
        out = fit_ea(self.score_weights,
                     self._model_selection,
                     self.param_distributions,
                     early_stop=self._model_selection['early_stop'],
                     toolbox=self._model_selection['toolbox'])
        self._pop, self._toolbox, self._ea_gen, self._evo_params = out

    def _as_dask_array(self, X, y=None, **kw):
        #if isinstance(self.estimator, Pipeline):
         #   self.estimator._run_generic_only = True
          #  X, y = self.estimator.fit_transform(X, y)
           # delattr(self.estimator, '_run_generic_only')
            #self.estimator._skip_generic = True
        if isinstance(X, (xr.Dataset, MLDataset)):
            X = MLDataset(X)
            if not X.has_features(raise_err=False):
                X = X.to_features()
        row_idx = getattr(X.features, X.features.dims[0])
        self.estimator._temp_row_idx = row_idx
        if isinstance(X, MLDataset):
            chunks = (int(DASK_CHUNK_N / X.features.shape[1]), 1)
            val = X.features.values
        else:
            chunks = (int(DASK_CHUNK_N / X.shape[1]), 1)
            val = X
        X = da.from_array(val, chunks=chunks)
        y = da.from_array(y, chunks=chunks[0]) # TODO
        return X, y

    def fit(self, X, y=None, groups=None, **fit_params):
        self._open()
        X, y = self._as_dask_array(X, y=y)
        for self._gen in range(self.ngen):
            print('gen', self._gen)
            DaskBaseSearchCV.fit(self, X, y, groups, **fit_params)
            fitnesses = self._get_cv_scores()
            self.cv_results_all_gen_ = _concat_cv_results(self.cv_results_all_gen_,
                                                          self.cv_results_,
                                                          gen=self._gen)
            if self._is_ea:
                out = self._ea_gen.send(fitnesses)
                self._pop, self._invalid_ind, self._param_history = out
                if not self._invalid_ind:
                    break
            else:
                self.next_params_ = self._model_selection(self.cv_results_all_gen_, X, y)
                if not self.next_params_:
                    break
        self._close()
        return self

    def _get_param_iterator(self):
        if hasattr(self, '_invalid_ind') and not self._invalid_ind:
            return iter(())
        return self._within_gen_param_iter(gen=self._gen)

    set_params = RandomizedSearchCV.set_params
    get_params = RandomizedSearchCV.get_params

