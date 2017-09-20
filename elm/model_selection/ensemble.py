from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import copy
from itertools import product

from dask_searchcv.model_selection import (GridSearchCV,
                                           _DOC_TEMPLATE,
                                           DaskBaseSearchCV)
from deap import base
from deap import creator
from deap import tools
import numpy as np
from sklearn.ensemble.base import BaseEnsemble
from sklearn import model_selection
from sklearn.utils import check_random_state
from sklearn.model_selection._search import _check_param_grid
from elm.model_selection.evolve import (fit_ea, DEFAULT_CONTROL,
                                        ind_to_new_params)
from elm.model_selection.ea_searchcv import EaSearchCV
from elm.model_selection.sorting import pareto_front

_ens_oneliner = '' # See GridSearchCV for ideas here - TODO
_ens_description = '' #TODO
_ens_parameters = ''#TODO
_ens_example = ''#TODO


class EnsembleCV(EaSearchCV, BaseEnsemble):
    __doc__ = _DOC_TEMPLATE.format(name="EnsembleCV",
                                   oneliner=_ens_oneliner,
                                   description=_ens_description,
                                   parameters=_ens_parameters,
                                   example=_ens_example)

    def __init__(self, estimator, estimators=None, n_estimators=10,
                 param_grid=None, ngen=3, estimator_params=tuple(),
                 score_weights=None, selection_func=None,
                 keep_n=None, init_n=None, sort_fitness=None, random_state=None,
                 scoring=None, refit=True, cv=None, error_score='raise',
                 iid=True, return_train_score=True, scheduler=None, n_jobs=-1,
                 cache_cv=True, select_with_test=True):

        if estimators:
            self.init_params_list = [est.get_params() for est in estimators]
            if n_estimators < len(self.init_params_list):
                n_estimators = len(self.init_params_list)
        else:
            self.init_params_list = []
        BaseEnsemble.__init__(self, base_estimator=estimator,
                              n_estimators=n_estimators,
                              estimator_params=tuple())
        DaskBaseSearchCV.__init__(self, estimator=estimator,
                scoring=scoring, iid=iid, refit=refit, cv=cv,
                error_score=error_score, return_train_score=return_train_score,
                scheduler=scheduler, n_jobs=n_jobs, cache_cv=cache_cv)
        self._switch_scheduler(self.estimator)
        self.ngen = ngen
        self.random_state = random_state
        self.param_grid = param_grid
        self.keep_n = keep_n
        self.init_n = init_n
        self.selection_func = selection_func
        self.score_weights = score_weights
        self.select_with_test = select_with_test
        self.sort_fitness = sort_fitness or pareto_front
        self.cv_results_all_gen_ = {}
        self._split_random_params()

    def _switch_scheduler(self, estimator):
        estimator.set_params(scheduler=self.scheduler)


    def _split_random_params(self):
        callable_params = {}
        choice_params = {}
        for name, params in self.param_grid.items():
            if hasattr(params, 'rvs'):
                callable_params[name] = params
            else:
                choice_params[name] = params
        if choice_params:
            _check_param_grid(choice_params)
        self.callable_params_ = callable_params
        self.choice_params_ = choice_params
        return callable_params, choice_params

    def _get_param_iterator(self, random_state=None):
        """Return ParameterSampler instance for the given distributions"""
        return model_selection.ParameterSampler(self.param_distributions,
                self.n_estimators, random_state=random_state)

    def _model_selection(self, fitnesses):
        best_idxes = self.sort_fitness(self.score_weights, np.array(fitnesses))
        if self.selection_func:
            kw = self.get_params()
            kw['gen'] = self._gen
            kw['rand_params'] = lambda: np.random.choice(self._select_models_default(fitnesses, best_idxes))
            out = list(self.selection_func(self.next_params_, fitnesses, best_idxes, **kw))
            params = [(x.get_params() if hasattr(x, 'get_params') else x)
                      for x in out]
        else:
            params = self._select_models_default(fitnesses, best_idxes)
        return params

    def _select_models_default(self, fitnesses, best_idxes):
        params = self.get_params()
        init_n, keep_n = params.get('init_n'), params.get('keep_n')
        if init_n:
            new = list(self._within_gen_param_iter(n_estimators=init_n))
        else:
            new = []
        if keep_n:
            next_params_ = copy.deepcopy(list(self.next_params_))
            keep = [next_params_[idx] for idx in best_idxes[:keep_n]]
        else:
            keep = []
        keep += new
        np.random.shuffle(keep)
        return keep

    def fit(self, X, y=None, groups=None, **fit_params):
        for self._gen in range(self.ngen):
            print('gen', self._gen)
            DaskBaseSearchCV.fit(self, X, y, groups, **fit_params)
            fitnesses = EaSearchCV._get_cv_scores(self)
            self.next_params_ = self._model_selection(fitnesses)
            self._concat_cv_results(self.cv_results_all_gen_,
                                    self.cv_results_,
                                    gen=self._gen)

        self.cv_results_ = self.cv_results_all_gen_
        delattr(self, 'cv_results_all_gen_')

    def _get_param_iterator_grid(self):
        """Return ParameterGrid instance for the given param_grid"""
        return model_selection.ParameterGrid(self.choice_params_)

    def _get_param_iterator_dist(self, n_estimators=None):
        """Return ParameterGrid instance for the given param_grid"""
        n_estimators = n_estimators or self.n_estimators
        return model_selection.ParameterSampler(self.callable_params_,
                n_estimators, random_state=self.random_state)

    def _within_gen_param_iter(self, gen=0, n_estimators=None):
        grid = self._get_param_iterator_grid()
        random_params = self._get_param_iterator_dist()
        grid, random_params = map(list, (grid, random_params))
        if n_estimators is None:
            n_estimators = self.n_estimators
        for idx in range(n_estimators):
            params = np.random.choice(grid)
            r = np.random.choice(random_params)
            params.update(r)
            yield params

    def _get_param_iterator(self):
        if self._gen == 0 and self.init_params_list:
            self.next_params_ = [self.estimator.get_params()] + self.init_params_list
        else:
            self.next_params_ = list(self._within_gen_param_iter(gen=self._gen))
        for params in self.next_params_:
            yield params

