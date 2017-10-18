from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import copy
from functools import partial

import dask.array as da
from dask_searchcv.model_selection import (_DOC_TEMPLATE,
                                           RandomizedSearchCV,
                                           DaskBaseSearchCV,
                                           _randomized_parameters)
import numpy as np
from elm.model_selection.evolve import (fit_ea,
                                        DEFAULT_CONTROL,
                                        ind_to_new_params,
                                        DEFAULT_EVO_PARAMS,)
from elm.mldataset.serialize_mixin import SerializeMixin
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
Evolutionary search (NSGA-2) for hyperparameterization or custom model selection
for batches of models in a generational process
"""
_ea_description = """\
The parameters of the estimator used to apply these methods are optimized
by cross-validated evolutionary algorithm search over a parameter grid.\
"""
_ea_parameters = _randomized_parameters + """\
ngen : Number of generations (each generation uses
    dask_searchcv.model_selection.RandomizedSearchCV)
score_weights : None if doing single objective minimization or a sequence of
    weights to use for flipping minimization to maximization, e.g.
    [1, -1, 1] would minimize the 1st and 3rd objectives and maximize the second
sort_fitness : Callable to be used for sorting model objective scores, by default
    it is elm.model_selection.sorting.pareto_front
model_selection : A callable that is called after each generation [TODO docs],
    or a dictionary of parameters passed to a generic evolutionary algorithm
    using Distributed Evolutionary Algorithms in Python (deap).  Here's an
    example of a dictionary passed to deap:
    model_selection = {
      'select_method': 'selNSGA2',
      'crossover_method': 'cxTwoPoint',
      'mutate_method': 'mutUniformInt',
      'init_pop': 'random',
      'indpb': 0.5,
      'mutpb': 0.9,
      'cxpb':  0.3,
      'eta':   20,
      'ngen':  2,
      'mu':    4,
      'k':     4,
      'early_stop': None

    }
model_selection_kwargs : Keyword arguments passed to the model selection
    callable (if given) otherwise ignored
select_with_test : Select / sort models based on test batch scores(True is default)
avoid_repeated_params : Avoid repeated parameters (True by default)
"""
_ea_example = """\
>>> from sklearn import svm, datasets
>>> from elm.model_selection.ea_searchcv import EaSearchCV
>>> from scipy.stats import lognorm
>>> from xarray_filters.datasets import make_classification
>>> X = make_classification()
>>> y = X.y.values.ravel()
>>> X = X.drop('y')
>>> parameters = {'kernel': ['linear', 'rbf'], 'C': lognorm(4)}
>>> svc = svm.SVC()
>>> clf = EaSearchCV(svc, parameters)
>>> clf.fit(X, y)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
EaSearchCV(avoid_repeated_params=True, cache_cv=True, cv=None,
      error_score='raise',
      estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False),
      iid=True, model_selection=None, model_selection_kwargs=None,
      n_iter=10, n_jobs=-1, ngen=3,
      param_distributions={'C': <scipy.stats._distn_infrastructure.rv_frozen object at 0x11d48f8d0>, 'kernel': ['linear', 'rbf']},
      random_state=None, refit=True, return_train_score=True,
      scheduler=None, score_weights=None, scoring=None,
      select_with_test=True, sort_fitness=None)
>>> sorted(clf.cv_results_.keys())  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
['gen', mean_fit_time', 'mean_score_time', 'mean_test_score',...
 'mean_train_score', 'param_C', 'param_kernel', 'params',...
 'rank_test_score', 'split0_test_score',...
 'split0_train_score', 'split1_test_score', 'split1_train_score',...
 'split2_test_score', 'split2_train_score',...
 'std_fit_time', 'std_score_time', 'std_test_score', 'std_train_score'...]\
"""

class EaSearchCV(RandomizedSearchCV, SklearnMixin, SerializeMixin):

    __doc__ = _DOC_TEMPLATE.format(name="EaSearchCV",
                                   oneliner=_ea_oneliner,
                                   description=_ea_description,
                                   parameters=_ea_parameters,
                                   example=_ea_example)

    def __init__(self, estimator, param_distributions, n_iter=10,
                 random_state=None,
                 ngen=3, score_weights=None,
                 sort_fitness=pareto_front,
                 model_selection=None,
                 model_selection_kwargs=None,
                 select_with_test=True,
                 avoid_repeated_params=True,
                 scoring=None,
                 iid=True, refit=True,
                 cv=None, error_score='raise', return_train_score=True,
                 scheduler=None, n_jobs=-1, cache_cv=True):
        filter_kw_and_run_init(RandomizedSearchCV.__init__, **locals())
        self.ngen = ngen
        self.select_with_test = select_with_test
        self.model_selection = model_selection
        self.model_selection_kwargs = model_selection_kwargs
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
        if callable(self._model_selection):
            return
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
        if isinstance(X, np.ndarray):
            return X, y
        if isinstance(X, (xr.Dataset, MLDataset)):
            X = MLDataset(X)
            if not X.has_features(raise_err=False):
                X = X.to_features()
        if not hasattr(X, 'features'):
            return X, y
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
            print('Generation', self._gen)
            RandomizedSearchCV.fit(self, X, y, groups, **fit_params)
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
                self.next_params_ = self._model_selection(self.next_params_,
                                                          np.array(fitnesses),
                                                          cv_results=self.cv_results_all_gen_,
                                                          X=X,
                                                          y=y)
                if not self.next_params_:
                    break
        self._close()
        return self

    def _get_param_iterator(self):
        if self._is_ea and not getattr(self, '_invalid_ind', None):
            return iter(())
        if not self._is_ea and self._gen == 0:
            self.next_params_ = tuple(RandomizedSearchCV._get_param_iterator(self))
        return self._within_gen_param_iter(gen=self._gen)

    set_params = RandomizedSearchCV.set_params
    get_params = RandomizedSearchCV.get_params

