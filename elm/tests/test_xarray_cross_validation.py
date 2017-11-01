from __future__ import print_function, unicode_literals, division

from collections import OrderedDict
import datetime

from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit
from xarray_filters import MLDataset
from xarray_filters.datasets import make_regression
from xarray_filters.pipeline import Generic, Step
import numpy as np
import pytest


from elm.mldataset import CV_CLASSES
from elm.model_selection import EaSearchCV
from elm.model_selection.sorting import pareto_front
from elm.pipeline import Pipeline
from elm.pipeline.predict_many import predict_many
from elm.pipeline.steps import linear_model,cluster
import elm.mldataset.cross_validation as cross_validation

START_DATE = datetime.datetime(2000, 1, 1, 0, 0, 0)
MAX_TIME_STEPS = 144
DATES = np.array([START_DATE - datetime.timedelta(hours=hr)
                 for hr in range(MAX_TIME_STEPS)])
DATE_GROUPS = np.linspace(0, 5, DATES.size).astype(np.int32)


# TODO - also test regressors
param_distributions = {
    'estimator__fit_intercept': [True, False],
}

param_distributions = {
    'estimator__n_clusters': [4,5,6,7,8, 10, 12],
    'estimator__init': ['k-means++', 'random'],
    'estimator__copy_x': [False],
    'estimator__algorithm': ["auto", "full", "auto"],
}

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
    dset = make_regression()
    dset.attrs['example_function_argument'] = date
    # TODO - this is not really testing
    # MLDataset as X because of .features.values below
    return dset.to_features(keep_attrs=True).features.values


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

pipe = Pipeline([ # TODO see note above about supervised models
    ('get_y', GetY()),
    ('estimator', linear_model.LinearRegression(n_jobs=-1)),
])

pipe = Pipeline([
    #('get_y', GetY()),  # TODO this wasn't working but should
    ('estimator', cluster.KMeans(n_jobs=1)),
])

@pytest.mark.parametrize('cls', CV_CLASSES)
def test_each_cv(cls):
    cv = getattr(cross_validation, cls)()
    ea = EaSearchCV(pipe,
                    param_distributions=param_distributions,
                    sampler=Sampler(),
                    ngen=2,
                    model_selection=model_selection,
                    cv=cv,
                    refit=False) # TODO refit = True

    print(ea.get_params())
    ea.fit(DATES, groups=DATE_GROUPS)
    results = getattr(ea, 'cv_results_', None)
    assert isinstance(results, dict) and 'gen' in results and all(getattr(v,'size',v) for v in results.values())

