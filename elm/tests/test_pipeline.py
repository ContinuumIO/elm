from argparse import Namespace
from collections import OrderedDict
from itertools import product

import sklearn.tree
from elm.pipeline import steps, Pipeline
from sklearn.base import ClassifierMixin
from sklearn.base import ClusterMixin
from sklearn.base import clone
from sklearn.ensemble.base import BaseEnsemble
from xarray_filters import MLDataset
import xarray_filters.datasets as xr_datasets
from xarray_filters.pipeline import Generic
from xarray_filters.tests.test_data import new_test_dataset
from xarray_filters.func_signatures import get_args_kwargs_defaults
import numpy as np
import pytest


ALL = {}
for m in dir(steps):
    sk_module = getattr(steps, m)
    if isinstance(sk_module, Namespace):
        for name, estimator in vars(sk_module).items():
            ALL[(m, name)] = estimator


REQUIRES_1D = ['IsotonicRegression']

XFAIL = ('label_propagation', 'semi_supervised', 'multiclass',
         'multioutput', 'ensemble', 'kernel_ridge', 'covariance',
         'naive_bayes', 'calibration', 'cross_decomposition', 'IsotonicRegression',
         'MultiTaskLassoCV', 'MultiTaskLasso',
         'MultiTaskElasticNetCV', 'MultiTaskElasticNet',
         'RANSACRegressor',
         'RFE', 'RFECV', 'Birch')
ITEMS = OrderedDict(sorted((k, v) for k, v in ALL.items()
                     if hasattr(v, '_cls') and
                     'fit' in dir(v._cls) and
                     'predict' in dir(v._cls)))
PREPROC = ['decomposition',
           'feature_selection',
           'manifold',
           'preprocessing',]

TRANSFORMERS = OrderedDict(sorted((k,v) for k, v in ALL.items() if k[0] in PREPROC))

SLOW = ('DictionaryLearning',)

shape = (8, 10)
def make_X_y(as_1d=False, is_classifier=False, is_cluster=False):
    layers = ['layer_{}'.format(idx) for idx in range(0, 10)]
    kw = dict(shape=shape, n_samples=np.prod(shape), layers=layers)
    if is_classifier:
        X = xr_datasets.make_classification(**kw)
        y = X.y.values.ravel()
        X.drop('y') # TODO handle in xarray_filters issue
    else:
        if is_cluster:
            X = xr_datasets.make_blobs(**kw)
        else:
            X = xr_datasets.make_regression(**kw)
        X2 = X.to_features()
        beta = np.random.uniform(0, 1, X2.features.shape[1])
        y = X2.features.values.dot(beta)
    if as_1d:
        X = _as_1d(X)
    return X, y


def _as_1d(X):
    return MLDataset(OrderedDict([('features', X.to_features().features[:, :1])]))


def get_params_for_est(estimator, name):
    is_classifier = ClassifierMixin in estimator.__mro__
    is_cluster = ClusterMixin in estimator.__mro__
    is_ensemble = BaseEnsemble in estimator.__mro__
    as_1d = name in REQUIRES_1D
    args, params, _ = get_args_kwargs_defaults(estimator.__init__)
    est_keys = (set(params) | set(args)) & set(('estimator', 'base_estimator', 'estimators'))
    for key in est_keys:
        if is_classifier:
            params[key] = sklearn.tree.DecisionTreeClassifier()
        else:
            params[key] = sklearn.tree.DecisionTreeRegressor()
        if key == 'estimators':
            params[key] = [(str(_), clone(params[key])) for _ in range(10)]
    X, y = make_X_y(as_1d=as_1d, is_classifier=is_classifier,
                    is_cluster=is_cluster)
    return X, y, params


def skipif(modul, name):
    if modul in XFAIL or name in XFAIL:
        pytest.xfail('{} - not implemented'.format(m if m in XFAIL else name))


@pytest.mark.parametrize('modul, name', ITEMS.keys())
def test_fit_predict_estimator(modul, name):
    try:
        estimator = ITEMS[(modul, name)]
        skipif(modul, name)
        X, y, params = get_params_for_est(estimator, name)
        mod = estimator(**params)
        fitted = mod.fit(X, y)
        assert isinstance(fitted, estimator)
        pred = fitted.predict(X)
        assert isinstance(pred, MLDataset)
        assert tuple(pred.data_vars) == ('predict',)
        # TODO - this should work assert tuple(pred.predict.dims) == tuple(X.dims)
    except:
        print(m, name, estimator)
        raise


def new_pipeline(*args):
    trans = []
    for idx, model in enumerate(args):
        name = model._cls.__name__.split('.')[-1]
        X, y, params = get_params_for_est(model, name)
        trans.append(('step_{}'.format(idx + 1), model(**params)))
    flatten = Generic(func=lambda X, y, **kw: X.to_features())
    trans = [('flat', flatten)] + trans
    pipe = Pipeline(trans)
    return pipe, X, y

pipe_combos = product(TRANSFORMERS.keys(), ITEMS.keys())
modules_names = [(k1, v1, k2, v2)
                 for (k1, v1), (k2, v2) in pipe_combos]
modules_names = [(item if not any(s in item for s in SLOW) else pytest.mark.slow(item))
                  for item in modules_names]
@pytest.mark.parametrize('m1, n1, m2, n2', modules_names)
def test_pipeline_combos(m1, n1, m2, n2):
    skipif(m1, n1)
    skipif(m2, n2)
    transformer = TRANSFORMERS[(m1, n1)]
    estimator = ITEMS[(m2, n2)]
    pipe, X, y = new_pipeline(transformer, estimator)
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert isinstance(pred, MLDataset)



