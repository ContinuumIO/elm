from __future__ import print_function, division
import dask
dask.set_options(get=dask.local.get_sync)
from collections import OrderedDict
import datetime
from itertools import product

from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit
from xarray_filters import MLDataset
from xarray_filters.datasets import make_regression
from xarray_filters.pipeline import Generic, Step
import numpy as np
import pytest


from elm.model_selection import EaSearchCV
from elm.model_selection.sorting import pareto_front
from elm.pipeline import Pipeline
from elm.model_selection import CVCacheSampler
from elm.pipeline.predict_many import predict_many
from elm.pipeline.steps import linear_model, cluster, decomposition
import sklearn.model_selection as sk_model_selection
from elm.tests.util import SKIP_CV, catch_warnings

START_DATE = datetime.datetime(2000, 1, 1, 0, 0, 0)
MAX_TIME_STEPS = 8
DATES = np.array([START_DATE - datetime.timedelta(hours=hr)
                 for hr in range(MAX_TIME_STEPS)])
DATE_GROUPS = np.linspace(0, 5, DATES.size).astype(np.int32)
'''
CV_CLASSES = dict([(k, getattr(sk_model_selection, k)) for k in dir(sk_model_selection)
              if isinstance(getattr(sk_model_selection, k), type) and
              issubclass(getattr(sk_model_selection, k),
                         sk_model_selection._split.BaseCrossValidator)])
CV_CLASSES.pop('BaseCrossValidator')
'''
CV_CLASSES = {'KFold': sk_model_selection.KFold}
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
    'mu':    16,
    'k':     8, # TODO ensure that k is not ignored - make elm issue if it is
    'early_stop': None
}

def example_function(date):
    dset = make_regression(n_samples=400,
                           layers=['layer_{}'.format(idx) for idx in range(5)])
    dset.attrs['example_function_argument'] = date
    return dset

class Sampler(Step):
    def transform(self, X, y=None, **kw):
        return example_function(X)


class GetY(Step):
    layer = 'y'
    def transform(self, X, y=None, **kw):
        layer = self.get_params()['layer']
        y = getattr(X, layer).values.ravel()
        X = MLDataset(OrderedDict([(k, v) for k, v in X.data_vars.items()
                                    if k != layer])).to_features()
        return X.features.values, y
    fit_transform = transform


# TODO - also test regressors
regress_distributions = {
    'estimator__fit_intercept': [True, False],
    'estimator__normalize': [True, False],
}

kmeans_distributions = {
    'estimator__n_clusters': list(range(4, 12)),
    'estimator__init': ['k-means++', 'random'],
    'estimator__copy_x': [False],
    'estimator__algorithm': ["auto", "full", "auto"],
}
pca_distributions = {
    'pca__n_components': list(range(2, 4)),
    'pca__whiten': [True, False],
}

regress = Pipeline([
    ('get_y', GetY()),
    ('estimator', linear_model.Ridge()),
])

pca_regress = Pipeline([
    ('get_y', GetY()),
    ('pca', decomposition.PCA()),
    ('estimator', linear_model.Ridge()),
])

kmeans = Pipeline([
    ('estimator', cluster.KMeans()),
])

configs = {'one_step_unsupervised': kmeans,
           'get_y_supervised':  regress,
           'get_y_pca_then_regress': pca_regress,}

dists = {'one_step_unsupervised': kmeans_distributions,
         'get_y_supervised': regress_distributions.copy(),
         'get_y_pca_then_regress': pca_distributions.copy(),}
dists['get_y_pca_then_regress'].update(regress_distributions)
refit_options = (False, True)
test_args = product(CV_CLASSES, configs, refit_options)
get_marks = lambda cls: [pytest.mark.slow] if cls.startswith(('Leave', 'Repeated')) else []
test_args = [pytest.param(c, key, refit, marks=get_marks(c))
             for c, key, refit in test_args]
@catch_warnings
@pytest.mark.parametrize('cls, config_key, refit', test_args)
def test_each_cv(cls, config_key, refit):
    if cls in SKIP_CV:
        pytest.skip('sklearn.model_selection cross validator {} is not yet supported'.format(cls))
    pipe = configs[config_key]
    param_distributions = dists[config_key]
    kw = dict()
    if cls.startswith('LeaveP'):
        kw['p'] = 2
    elif cls == 'PredefinedSplit':
        kw['test_fold'] = (DATES > DATES[DATES.size // 2]).astype(np.int32)
    cv = CV_CLASSES[cls](**kw)
    sampler = Sampler()
    refit_Xy = sampler.fit_transform([datetime.datetime(2000, 1, 1)])
    refit = True
    ea = EaSearchCV(pipe,
                    param_distributions=param_distributions,
                    sampler=sampler,
                    ngen=2,
                    model_selection=model_selection,
                    cv=cv,
                    refit=refit,
                    refit_Xy=refit_Xy)
    ea.fit(DATES) # TODO test that y is passed as a cv grouping variable
    results = getattr(ea, 'cv_results_', None)
    assert isinstance(results, dict) and 'gen' in results
    assert np.unique([getattr(v, 'size', len(v)) for v in results.values()]).size == 1

