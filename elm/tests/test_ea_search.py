from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import product
import os

from dask_glm.datasets import make_classification
from sklearn import decomposition as sk_decomp
from sklearn import svm as sk_svm
from sklearn.pipeline import Pipeline as sk_Pipeline
from xarray_filters import MLDataset
from xarray_filters.datasets import _make_base
import dill
import numpy as np
import pandas as pd
import pytest
import xarray as xr


from elm.mldataset.wrap_sklearn import (_as_numpy_arrs,
                                        _from_numpy_arrs)
from elm.model_selection.ea_searchcv import EaSearchCV
from elm.model_selection.multilayer import MultiLayer
from elm.pipeline import Pipeline
from elm.pipeline.steps import (linear_model as lm,
                                preprocessing as elm_pre,
                                decomposition as elm_decomp,
                                svm as elm_svm,)
from elm.tests.test_pipeline import new_pipeline, modules_names
from elm.tests.util import (TRANSFORMERS, TESTED_ESTIMATORS,
                            catch_warnings, skip_transformer_estimator_combo,
                            make_X_y)

param_distribution_poly = dict(step_1__degree=list(range(1, 3)),
                               step_1__interaction_only=[True, False])
param_distribution_pca = dict(step_1__n_components=list(range(1, 12)),
                              step_1__whiten=[True, False])
param_distribution_sgd = dict(step_2__penalty=['l1', 'l2', 'elasticnet'],
                              step_2__alpha=np.logspace(-1, 1, 5))

model_selection = dict(mu=16,       # Population size
                       ngen=3,      # Number of generations
                       mutpb=0.4,   # Mutation probability
                       cxpb=0.6,    # Cross over probability
                       param_grid_name='example_1') # CSV based name for parameter / objectives history

def make_choice(ea):
    num = np.random.randint(1, len(ea) + 1)
    idx = np.random.randint(0, len(ea), (num,))
    return [ea[i] for i in idx]


zipped = product((elm_pre.PolynomialFeatures, elm_decomp.PCA),
                 (lm.SGDRegressor,),)
tested_pipes = [(trans, estimator)
                for trans, estimator in zipped]
@catch_warnings
@pytest.mark.parametrize('trans, estimator', tested_pipes)
def test_cv_splitting_ea_search_mldataset(trans, estimator):
    '''Test that an Elm Pipeline using MLDataset X feature
    matrix input can be split into cross validation train / test
    samples as in scikit-learn for numpy.  (As of PR 192 this test
    is failing)'''
    pipe, X, y = new_pipeline(trans, estimator, flatten_first=False)
    X = X.to_features()
    param_distribution = param_distribution_sgd.copy()
    if 'PCA' in trans._cls.__name__:
        param_distribution.update(param_distribution_pca)
    else:
        param_distribution.update(param_distribution_poly)
    ea = EaSearchCV(estimator=pipe,
                    param_distributions=param_distribution,
                    score_weights=[1],
                    model_selection=model_selection,
                    refit=True,
                    cv=3,
                    error_score='raise',
                    return_train_score=True,
                    scheduler=None,
                    n_jobs=-1,
                    cache_cv=True)
    ea.fit(X,y)
    assert isinstance(ea.predict(X), MLDataset)


def make_dask_arrs():
    return make_classification(n_samples=300, n_features=6)

def make_np_arrs():
    return [_.compute() for _ in make_dask_arrs()]

def make_dataset(flatten_first=True):
    X, y = make_mldataset(flatten_first=flatten_first)
    return xr.Dataset(X), y

def make_mldataset(flatten_first=True):
    X, y = make_X_y(astype='MLDataset', is_classifier=True,
                    flatten_first=flatten_first)
    return X, y

def make_dataframe():
    X, y = make_np_arrs()
    X = pd.DataFrame(X)
    return X, y

def model_selection_example(params_list, best_idxes, **kw):
    top_n = kw['top_n']
    new = len(params_list) - top_n
    params = [params_list[idx] for idx in best_idxes[:top_n]]
    new = [dict(C=parameters['C'].rvs(), kernel='linear')
           for _ in range(new)]
    return params + new

data_structure_trials = [('pandas', make_dataframe),
                         ('dataset', make_dataset),
                         ('dask.dataframe', make_dask_arrs),
                         ('mldataset', make_mldataset),
                         ('numpy', make_np_arrs),]

model_sel_kwargs = [None, dict(top_n=4)]
model_sel = [None, model_selection_example]

args = {}
for label, make_data in data_structure_trials:
    if label in ('numpy', 'pandas', 'dask.dataframe'):
        est = sk_svm.SVC()
        trans = sk_decomp.PCA(n_components=2)
    else:
        est = elm_svm.SVC()
        trans = elm_decomp.PCA(n_components=2)
    for s in ([('trans', trans), ('est', est)], [('est', est,),], []):
        pipe_cls = sk_Pipeline, Pipeline
        pipe_word = 'sklearn.pipeline', 'elm.pipeline'
        for cls, word in zip(pipe_cls, pipe_word):
            if s:
                est = cls(s)
                label2 = 'PCA-SVC-{}'
            else:
                label2 = 'SVC-{}'
            for sel, kw in zip(model_sel, model_sel_kwargs):
                args[label + '-' + label2.format(word)] = (est, make_data, sel, kw)


@pytest.mark.parametrize('label, do_predict', product(args, (True, False)))
def test_ea_search_sklearn_elm_steps(label, do_predict):
    '''Test that EaSearchCV can work with numpy, dask.array,
    pandas.DataFrame, xarray.Dataset, xarray_filters.MLDataset
    '''
    from scipy.stats import lognorm
    est, make_data, sel, kw = args[label]
    parameters = {'kernel': ['linear', 'rbf'],
                  'C': lognorm(4),}
    if isinstance(est, (sk_Pipeline, Pipeline)):
        parameters = {'est__{}'.format(k): v
                      for k, v in parameters.items()}
    ea = EaSearchCV(est, parameters,
                    n_iter=4,
                    ngen=2,
                    model_selection=sel,
                    model_selection_kwargs=kw)
    X, y = make_data()
    ea.fit(X, y)
    if do_predict:
        pred = ea.predict(X)
        assert isinstance(pred, type(y))

