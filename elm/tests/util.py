from __future__ import absolute_import, division, print_function, unicode_literals
from argparse import Namespace
from collections import OrderedDict
from functools import wraps
import os
import warnings

import pytest
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.base import ClusterMixin
from sklearn.ensemble.base import BaseEnsemble
from sklearn.exceptions import ConvergenceWarning
import sklearn.feature_selection as feat
import sklearn.tree
from xarray_filters import MLDataset
import numpy as np
import xarray_filters.datasets as xr_datasets
from xarray_filters.func_signatures import get_args_kwargs_defaults
import yaml

from elm.pipeline import steps

YAML_TEST_CONFIG = os.path.join(os.path.dirname(__file__), 'test_config.yaml')

with open(YAML_TEST_CONFIG) as f:
    contents = f.read()
TEST_CONFIG = yaml.safe_load(contents)

ALL_STEPS = steps.ALL_STEPS

REQUIRES_1D = ['IsotonicRegression']

SKIP = TEST_CONFIG['SKIP'] # TODO - See related skip_transformer_estimator_combo notes
SKIP_CV = TEST_CONFIG['SKIP_CV']
TESTED_ESTIMATORS = OrderedDict(sorted((k, v) for k, v in ALL_STEPS.items()
                     if hasattr(v, '_cls') and
                     'fit' in dir(v._cls) and
                     'predict' in dir(v._cls)))
PREPROC = ['decomposition',
           'feature_selection',
           'manifold',
           'preprocessing',]

TRANSFORMERS = OrderedDict(sorted((k,v) for k, v in ALL_STEPS.items() if k[0] in PREPROC))

SLOW = ('DictionaryLearning', 'MiniBatchDictionaryLearning')

USES_COUNTS = ('LatentDirichletAllocation', 'NMF')

def catch_warnings(func):
    '''Silence all warnings in tests
    TODO - See related skip_transformer_estimator_combo notes'''
    @wraps(func)
    def new_func(*args, **kw):
        skipped_warnings = (FutureWarning, UserWarning,
                            DeprecationWarning, ConvergenceWarning)
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore",
                                  category=skipped_warnings)
            return func(*args, **kw)
    return new_func


shape = (8, 10)
def make_X_y(astype='MLDataset', as_1d=False, is_classifier=False,
             is_cluster=False, uses_counts=False, **kw):
    '''Make X and y (as needed) for testing considering
    whether the context is for a classifier or clustering estimator
    or uses what are typically bin counts (uses_counts - LDA as an example)
    '''
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
    if uses_counts:
        # Since small data sets are used for unit test,
        # prevent classes who have sample counts of <3
        # (this prevents covariance failure that
        #  arises when a class membership is 1)
        for arr in X.data_vars.values():
            val = arr.values
            val[:] = np.round(np.abs(val)).astype(np.int32)
            uniq = np.unique(val.ravel())
            last_x = uniq[0]
            for xi in uniq:
                if val[val == xi].size < 3:
                    val[val == xi] = last_x
                last_x = xi
    if astype == 'numpy':
        X = X.to_features().features.values
    return X, y


def _as_1d(X):
    '''Some estimators require a 1-D X feature matrix'''
    return MLDataset(OrderedDict([('features', X.to_features().features[:, :1])]))


def get_params_for_est(estimator, name):
    '''Choose initialization parameters for an estimator for auto-testing'''
    is_classifier = ClassifierMixin in estimator.__mro__
    is_cluster = ClusterMixin in estimator.__mro__
    is_ensemble = BaseEnsemble in estimator.__mro__
    uses_counts = any(c in name for c in USES_COUNTS)
    as_1d = name in REQUIRES_1D
    args, params, _ = get_args_kwargs_defaults(estimator.__init__)
    est_keys = set(('estimator', 'base_estimator', 'estimators'))
    est_keys = (set(params) | set(args)) & est_keys
    if is_classifier:
        score_func = feat.f_classif
    else:
        score_func = feat.f_regression
    for key in est_keys:
        if name == 'SelectFromModel':
            params[key] = sklearn.linear_model.LassoCV()
        elif is_classifier:
            params[key] = sklearn.tree.DecisionTreeClassifier()
        else:
            params[key] = sklearn.tree.DecisionTreeRegressor()
        if key == 'estimators':
            params[key] = [(str(_), clone(params[key])) for _ in range(10)]
    kw = dict(is_classifier=is_classifier, is_cluster=is_cluster,
              is_ensemble=is_ensemble, uses_counts=uses_counts)
    if 'score_func' in params:
        params['score_func'] = score_func
    X, y = make_X_y(**kw)
    return X, y, params, kw


def skip_transformer_estimator_combo(module1, cls_name1, module2, cls_name2):
    '''Skip a given transformer - estimator combination

    Parameters
    ----------
    module1: str - transformer module first step e.g. "decomposition"
    module2: str - estimator module second step e.g. "cluster"
    cls_name1: str - estimator class name, e.g. PCA
    cls_name2: str - estimator class name, e.g. KMeans

    Returns
    -------

    Returns True/False - TODO - Note we need to review each combo
    of transformer / estimator being skipped here and see if that is
    1) elm/xarray_filters library code deficiency,
    2) a test harness problem, e.g. the transformer needs an initalization
       argument that we have not put into the auto-testing harness here yet
    3) a combination of scikit-learn transformer and estimator that does
       not make sense in any context, e.g. PCA then IsotonicRegression
       (IsotonicRegression expects a 1-D X and 1-D y)
    4) some mix of items 1, 2, and 3

    The bullet comments 1 to 4 apply to most of the test skipping
    and warning silencing logic in elm.tests now.
    '''
    if module1 == module2:
        return
    skip = False
    if module2 in PREPROC:
        skip = True
    if module2 == 'ensemble': # Needs a regressor/classifier not a preproc
        skip = True
    for item in (module1, cls_name1, module2, cls_name2):
        if item in SKIP:
            skip = True
    if cls_name1 == 'NMF' and cls_name2 == 'RadiusNeighborsClassifier':
        skip = True
    elif cls_name1 == 'MDS' and cls_name2 == 'RadiusNeighborsClassifier':
        skip = True
    elif cls_name1 == 'SpectralEmbedding' and cls_name2 == 'GaussianProcess':
        skip = True
    elif cls_name1 == 'TSNE' and cls_name2 == 'GaussianProcess':
        skip = True
    elif cls_name1 == 'TSNE' and cls_name2 == 'RadiusNeighborsClassifier':
        skip = True
    elif cls_name1 =='KernelCenterer' or cls_name1 == 'MultiLabelBinarizer':
        skip = True
    elif cls_name1 == 'Binarizer' and cls_name2 == 'GaussianProcess':
        skip = True
    elif module1 in ('manifold', 'preprocessing', 'feature_selection', 'decomposition') and 'ensemble' == module2:
        skip = True
    return skip
