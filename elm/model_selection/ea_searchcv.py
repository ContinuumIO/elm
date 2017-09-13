from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import copy

from dask_searchcv.model_selection import DaskBaseSearchCV, _DOC_TEMPLATE
from deap import base
from deap import creator
from deap import tools
import numpy as np
from sklearn.model_selection._search import _check_param_grid
from elm.model_selection.evolve import (fit_ea, DEFAULT_CONTROL,
                                        ind_to_new_params)

_ea_oneliner = """\
Exhaustive search over specified parameter values for an estimator.\
"""
_ea_description = """\
The parameters of the estimator used to apply these methods are optimized
by cross-validated evolutionary algorithm search over a parameter grid.\
"""
_ea_parameters = """\
param_grid : dict or list of dictionaries
    Dictionary with parameters names (string) as keys and lists of
    parameter settings to try as values, or a list of such
    dictionaries, in which case the grids spanned by each dictionary
    in the list are explored. This enables searching over any sequence
    of parameter settings.\
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
        iid=..., n_jobs=..., param_grid=..., refit=..., return_train_score=...,
        scheduler=..., scoring=...)
>>> sorted(clf.cv_results_.keys())  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
['mean_fit_time', 'mean_score_time', 'mean_test_score',...
 'mean_train_score', 'param_C', 'param_kernel', 'params',...
 'rank_test_score', 'split0_test_score',...
 'split0_train_score', 'split1_test_score', 'split1_train_score',...
 'split2_test_score', 'split2_train_score',...
 'std_fit_time', 'std_score_time', 'std_test_score', 'std_train_score'...]\
"""

class EaSearchCV(DaskBaseSearchCV):
    __doc__ = _DOC_TEMPLATE.format(name="EaSearchCV",
                                   oneliner=_ea_oneliner,
                                   description=_ea_description,
                                   parameters=_ea_parameters,
                                   example=_ea_example)

    def __init__(self, estimator, param_grid,
                 score_weights, k=16, mu=32,
                 ngen=3, cxpb=0.3, indpb=0.5, mutpb=0.9, eta=20,
                 param_grid_name='param_grid',
                 early_stop=None, toolbox=None, scoring=None,
                 refit=True, cv=None, error_score='raise', iid=True,
                 return_train_score=True, scheduler=None, n_jobs=-1,
                 cache_cv=True, select_with_test=True):
        super(EaSearchCV, self).__init__(estimator=estimator,
                scoring=scoring, iid=iid, refit=refit, cv=cv,
                error_score=error_score, return_train_score=return_train_score,
                scheduler=scheduler, n_jobs=n_jobs, cache_cv=cache_cv)


        self.param_grid = param_grid
        self.estimator = estimator
        self.score_weights = score_weights
        self.k = k
        self.mu = mu
        self.indpb = indpb
        self.eta = eta
        self.ngen = ngen
        self.cxpb = cxpb
        self.early_stop = early_stop
        self.param_grid_name = param_grid_name
        self.select_with_test = select_with_test
        self.toolbox = toolbox
        self.cv_results_all_gen_ = {}
        _check_param_grid(self.param_grid)
        out = fit_ea(self.score_weights,
                     self._make_control_kw(),
                     self.param_grid,
                     early_stop=self.early_stop,
                     toolbox=toolbox)
        self._pop, self._toolbox, self._ea_gen, self._evo_params = out

    def _make_control_kw(self):
        control = {}
        for k, v in DEFAULT_CONTROL.items():
            control[k] = getattr(self, k, v)
        return control

    def _within_gen_param_iter(self, gen=0):
        deap_params = self._evo_params['deap_params']
        if gen == 0:
            invalid_ind = self._pop
        else:
            invalid_ind = self._invalid_ind
        for idx, ind in enumerate(invalid_ind):
            yield ind_to_new_params(deap_params, ind)

    def _concat_cv_results(self, cv1, cv2):
        cv_results = {}
        if not cv1:
            return cv2
        for k, v2 in cv2.items():
            v1 = cv1[k]
            if isinstance(v1, list):
                v = v1 + v2
            elif isinstance(v1, np.ndarray):
                v = np.concatenate((v1, v2)).squeeze()
            else:
                raise NotImplementedError('{} not handled ({})'.format(k, v1))
            cv_results[k] = v
        return cv_results

    def _get_cv_scores(self):
        cv_results = getattr(self, 'cv_results_', None)
        if not cv_results:
            raise ValueError('Expected "cv_results_" attribute')
        if self.select_with_test:
            score_field = 'mean_test_score'
        else:
            score_field = 'mean_train_score'
        return cv_results[score_field]

    def fit(self, X, y=None, groups=None, **fit_params):
        for gen in range(self.ngen):
            self.gen = gen
            super(EaSearchCV, self).fit(X, y, groups, **fit_params)
            fitnesses = self._get_cv_scores()
            if isinstance(fitnesses, np.ndarray) and fitnesses.squeeze().ndim == 1:
                fitnesses = [(x,) for x in fitnesses]
            else:
                raise NotImplementedError('Multi-objective optimization would require a few changes to cv_results?')
            out = self._ea_gen.send(fitnesses)
            self._pop, self._invalid_ind, self._param_history = out
            cv1, cv2 = self.cv_results_all_gen_, self.cv_results_
            self.cv_results_all_gen_ = self._concat_cv_results(cv1, cv2)
            if not self._invalid_ind:
                break
        self.cv_results_ = self.cv_results_all_gen_

    def _get_param_iterator(self):
        return self._within_gen_param_iter(gen=self.gen)
