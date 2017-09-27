from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
from itertools import product
import os

from xarray_filters import MLDataset
from xarray_filters.datasets import _make_base
import dill
import numpy as np
import pandas as pd
import pytest

from elm.mldataset.wrap_sklearn import (_as_numpy_arrs,
                                        _from_numpy_arrs)
from elm.model_selection.ea_searchcv import EaSearchCV
from elm.model_selection.multilayer import MultiLayer
from elm.pipeline import Pipeline
from elm.pipeline.steps import (linear_model as lm,
                                preprocessing as pre,
                                decomposition as decomp)
from elm.tests.test_pipeline import pipeline_xfail, new_pipeline, modules_names
from elm.tests.util import TRANSFORMERS, TESTED

param_distribution_poly = dict(step_1__degree=list(range(1, 3)),
                               step_1__interaction_only=[True, False])
param_distribution_pca = dict(step_1__n_components=list(range(1, 12)),
                              step_1__whiten=[True, False])

param_distribution_sgd = dict(step_2__penalty=['l1', 'l2', 'elasticnet'],
                           step_2__alpha=np.logspace(-4, 1, 5))

model_selection = dict(mu=16,       # Population size
                       ngen=3,      # Number of generations
                       mutpb=0.4,   # Mutation probability
                       cxpb=0.6,    # Cross over probability
                       param_grid_name='example_1') # CSV based name for parameter / objectives history



def fit_est(param_distribution, est, X, y, astype):

    ea = EaSearchCV(estimator=est,
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
    if astype == 'numpy':
        # TODO convert to numpy and test that way
        #pytest.xfail('')
        X = X.to_features().features.values
    return ea.fit(X, y=y)

def make_choice(ea):
    num = np.random.randint(1, len(ea) + 1)
    idx = np.random.randint(0, len(ea), (num,))
    return [ea[i] for i in idx]


zipped = product((pre.PolynomialFeatures, decomp.PCA),
                 (lm.SGDRegressor,),
                 ('numpy', 'mldataset'))
tested_pipes = [(trans, estimator, typ)
                for trans, estimator, typ in zipped]

@pytest.mark.parametrize('trans, estimator, astype', tested_pipes)
def test_ea_search(trans, estimator, astype):
    if astype == 'numpy':
        pytest.xfail('numpy in pipeline / EA search')
    pipe, X, y = new_pipeline(trans, estimator)
    param_distribution = param_distribution_sgd.copy()
    if 'PCA' in estimator.__name__:
        param_distribution.update(param_distribution_pca)
    else:
        param_distribution.update(param_distribution_poly)
    ea = fit_est(param_distribution, pipe, X, y, astype)
    assert isinstance(ea.predict(X), MLDataset)

