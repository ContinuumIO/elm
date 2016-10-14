import copy
from itertools import product
import os

from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_selection import f_classif
import numpy as np
import pytest

from elm.config import parse_env_vars, client_context
from elm.readers import *
# Below "steps" is a module of all the
# classes which can be used for Pipeline steps
from elm.pipeline import Pipeline, steps
from elm.pipeline.tests.util import random_elm_store
from elm.model_selection.kmeans import kmeans_model_averaging, kmeans_aic

LARGE_ENSEMBLE_KWARGS = dict(ngen=3, init_ensemble_size=25,
                       partial_fit_batches=3)

SMALL_ENSEMBLE_KWARGS = dict(ngen=2, init_ensemble_size=2,
                             partial_fit_batches=None)

if parse_env_vars()['ELM_LARGE_TEST']:
    ENSEMBLE_KWARGS = LARGE_ENSEMBLE_KWARGS
else:
    ENSEMBLE_KWARGS = SMALL_ENSEMBLE_KWARGS
EX1 = Pipeline([steps.Flatten('C'),
                steps.Transform(IncrementalPCA(n_components=3), partial_fit_batches=None),
                SGDClassifier()])
# note partial fit within the pipeline
EX2 = Pipeline([steps.Flatten('C'),
                steps.Transform(estimator=IncrementalPCA(n_components=3),
                                partial_fit_batches=None),
                MiniBatchKMeans(n_clusters=4)])


def example_sampler(h, w, bands, **kwargs):
    bands = ['band_{}'.format(idx + 1) for idx in range(bands)]
    return random_elm_store(width=w, height=h, bands=bands)

def example_get_y(X, y=None, sample_weight=None, **kwargs):
    '''Always return X,y, sample_weight'''
    y = MiniBatchKMeans(n_clusters=3).fit(X.flat.values).predict(X.flat.values)
    return (X, y, sample_weight)
# The following uses "ModifySample" to return X, y for next step
EX4 = Pipeline([steps.Flatten('C'),
                steps.ModifySample(example_get_y),
                steps.SelectPercentile(score_func=f_classif, percentile=50),
                SGDClassifier()])


def test_simple():
    p = Pipeline([steps.Flatten('C'), MiniBatchKMeans(n_clusters=5),])
    args_list = [(100, 200, 5)] * 10 # (height, width, bands)
    data_source = dict(sampler=example_sampler, args_list=args_list)
    ensemble_kw = dict(ngen=2, init_ensemble_size=2)
    fitted = p.fit_ensemble(**data_source, **ensemble_kw)
    tagged_fitted_models = fitted.ensemble
    (tag1, model1), (tag2, model2) = tagged_fitted_models # ensemble size of 2 here
    X = example_sampler(100, 400, 5)
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)
    assert pred1.shape == pred2.shape == (400 * 100,)

args_list = [(300, 200, 8)] * 10 # (height, width, bands) 10 samples
SAMPLER_DATA_SOURCE = dict(sampler=example_sampler, args_list=args_list)



X = example_sampler(600, 600, 8)
X_Y_DATA_SOURCE = {'X': X, 'y':example_get_y(flatten(X))[1]}

DATA_SOURCES = [SAMPLER_DATA_SOURCE, X_Y_DATA_SOURCE]
def tst_ensemble_scenarios(scenario, supervised, pipe):

        data_source = scenario(supervised=supervised)
        ensemble_kw = dict(client=client, **ENSEMBLE_KWARGS)
        fitted = copy.deepcopy(pipe).fit_ensemble(**data_source, **ensemble_kw)
        pred = fitted.predict_many(**data_source)

def dist_test(func):
    with client_context() as client: # taking dask env variables
        def new(*a, **kw):
            kw = dict(client=client, **kw)
            return func(*a, **kw)
        return new


def _train_asserts(fitted):
    ens = fitted.ensemble
    assert len(fitted.ensemble) == ENSEMBLE_KWARGS['init_ensemble_size']
    assert all(isinstance(x, tuple) and len(x) == 2 for x in ens)
    assert all(isinstance(x[1], Pipeline) for x in ens)
    assert all(isinstance(x[0], str) and x[0] for x in ens)


@dist_test
def test_kmeans_simple_sampler(client=None):
    pipe = Pipeline([steps.Flatten('C'),
                     MiniBatchKMeans(n_clusters=6)])
    fitted = pipe.fit_ensemble(**SAMPLER_DATA_SOURCE, **ENSEMBLE_KWARGS)
    ens = fitted.ensemble
    _train_asserts(fitted)
    pred = fitted.predict_many(**SAMPLER_DATA_SOURCE)
    assert len(pred) == len(SAMPLER_DATA_SOURCE['args_list']) * len(ens)


@dist_test
def test_kmeans_simple_X(client=None):
    pipe = Pipeline([steps.Flatten('C'),
                     MiniBatchKMeans(n_clusters=6)])
    fitted = pipe.fit_ensemble(X=X, **ENSEMBLE_KWARGS)
    _train_asserts(fitted)
    pred = fitted.predict_many(X=X)
    assert len(pred) == len(fitted.ensemble)


@dist_test
def test_supervised_feat_select_sampler(client=None):
    '''EX4 has a ModifySample step to get necessary y data'''

    pipe = Pipeline([steps.Flatten('C'),
                steps.ModifySample(example_get_y),
                steps.SelectPercentile(score_func=f_classif, percentile=50),
                SGDClassifier()])
    fitted = pipe.fit_ensemble(**SAMPLER_DATA_SOURCE, **ENSEMBLE_KWARGS)
    ens = fitted.ensemble
    _train_asserts(fitted)
    pred = fitted.predict_many(**SAMPLER_DATA_SOURCE)
    assert len(pred) == len(SAMPLER_DATA_SOURCE['args_list']) * len(ens)


@dist_test
def test_supervised_feat_select_X_y(client=None):
    '''EX4 has a ModifySample step to get necessary y data'''
    pipe = Pipeline([steps.Flatten('C'),
            steps.SelectPercentile(score_func=f_classif, percentile=50),
            SGDClassifier()])
    fitted = pipe.fit_ensemble(**X_Y_DATA_SOURCE, **ENSEMBLE_KWARGS)
    _train_asserts(fitted)
    pred = fitted.predict_many(**X_Y_DATA_SOURCE)
    assert len(pred) == len(fitted.ensemble)


@dist_test
def test_kmeans_model_selection(client=None):
    pipe = Pipeline([steps.Flatten('C'),
                    ('kmeans', MiniBatchKMeans(n_clusters=5))],
                    scoring=kmeans_aic,
                    scoring_kwargs={'score_weights': [-1]})
    en = ENSEMBLE_KWARGS.copy()
    choices = {'kmeans__n_clusters': list(range(3, 10))}
    def init(pipe, **kwargs):
        estimators = []
        for _ in range(24):
            n_clusters = np.random.choice(choices['kmeans__n_clusters'])
            estimator = copy.deepcopy(pipe)
            estimator.set_params(kmeans__n_clusters=n_clusters)
            estimators.append(estimator)
        return estimators

    en['ensemble_init_func'] = init
    en['model_selection_kwargs'] = dict(drop_n=8, evolve_n=16,
                                        init_n=8, choices=choices)
    en['model_selection'] = kmeans_model_averaging
    fitted = pipe.fit_ensemble(**SAMPLER_DATA_SOURCE, **en)
    assert len(fitted.ensemble) == 24
    preds = fitted.predict_many(**SAMPLER_DATA_SOURCE)
    assert len(preds) == len(fitted.ensemble) * len(SAMPLER_DATA_SOURCE['args_list'])
