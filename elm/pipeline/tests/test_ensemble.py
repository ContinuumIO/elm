'''

Tests usages like:

EX1 = Pipeline([steps.Flatten(),
                steps.Transform(IncrementalPCA(n_components=3), partial_fit_batches=None),
                SGDClassifier()])
EX2 = Pipeline([steps.Flatten(),
                steps.Transform(estimator=IncrementalPCA(n_components=3),
                                partial_fit_batches=None),
                MiniBatchKMeans(n_clusters=4)])


'''
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
from elm.sample_util.elm_store import ElmStore
from elm.sample_util.reshape import flatten
# Below "steps" is a module of all the
# classes which can be used for Pipeline steps
from elm.pipeline import Pipeline, steps
from elm.sample_util.make_blobs import random_elm_store
from elm.model_selection.kmeans import kmeans_model_averaging, kmeans_aic

ENSEMBLE_KWARGS = dict(ngen=2, init_ensemble_size=2,
                       saved_ensemble_size=2,
                       partial_fit_batches=2)



def example_sampler(h, w, bands, **kwargs):
    '''A sampler takes one of the elements of "args_list" to return X
       Alternatively a sampler taking an element of "args_list" can
       return (X, y, sample_weight) tuple
    '''
    bands = ['band_{}'.format(idx + 1) for idx in range(bands)]
    return random_elm_store(width=w, height=h, bands=bands)

def example_get_y(X, y=None, sample_weight=None, **kwargs):
    '''Always return X,y, sample_weight

    This function calculates a new y column numpy array
    and passes through the X and sample_weight given.
    Every user-given function in Pipeline must return
    (X, y, sample_weight) tuple'''
    y = MiniBatchKMeans(n_clusters=3).fit(X.flat.values).predict(X.flat.values)
    return (X, y, sample_weight)

def test_simple():
    p = Pipeline([steps.Flatten(), MiniBatchKMeans(n_clusters=5),])
    args_list = [(100, 200, 5)] * 10 # (height, width, bands)
    data_source = dict(sampler=example_sampler, args_list=args_list)
    ensemble_kw = dict(ngen=2, init_ensemble_size=2)
    ensemble_kw.update(data_source)
    fitted = p.fit_ensemble(**ensemble_kw)
    tagged_fitted_models = fitted.ensemble
    (tag1, model1), (tag2, model2) = tagged_fitted_models # ensemble size of 2 here
    X = example_sampler(100, 400, 5)
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)
    assert pred1.shape == pred2.shape == (400 * 100,)

args_list = [(300, 200, 8)] * 2 # (height, width, bands) 10 samples
SAMPLER_DATA_SOURCE = dict(sampler=example_sampler, args_list=args_list)

X = example_sampler(600, 600, 8)
X_Y_DATA_SOURCE = {'X': X, 'y':example_get_y(flatten(X))[1]}

DATA_SOURCES = [SAMPLER_DATA_SOURCE, X_Y_DATA_SOURCE]

def dist_test(func):
    with client_context() as client: # taking dask env variables
        def new(*a, **kw):
            kw = dict(client=client, **kw)
            return func(*a, **kw)
        return new


def _train_asserts(fitted, expected_len):
    ens = fitted.ensemble
    assert all(isinstance(x, tuple) and len(x) == 2 for x in ens)
    assert all(isinstance(x[1], Pipeline) for x in ens)
    assert all(isinstance(x[0], str) and x[0] for x in ens)
    assert len(fitted.ensemble) == expected_len


@dist_test
def test_kmeans_simple_sampler(client=None):
    pipe = Pipeline([steps.Flatten(),
                     MiniBatchKMeans(n_clusters=6)])
    kw = SAMPLER_DATA_SOURCE.copy()
    kw.update(ENSEMBLE_KWARGS)
    fitted = pipe.fit_ensemble(**kw)
    ens = fitted.ensemble
    _train_asserts(fitted, ENSEMBLE_KWARGS['saved_ensemble_size'])
    pred = fitted.predict_many(**SAMPLER_DATA_SOURCE)
    assert len(pred) == len(SAMPLER_DATA_SOURCE['args_list']) * len(ens)


@dist_test
def test_kmeans_simple_X(client=None):
    pipe = Pipeline([steps.Flatten(),
                     MiniBatchKMeans(n_clusters=6)])
    fitted = pipe.fit_ensemble(X=X, **ENSEMBLE_KWARGS)
    _train_asserts(fitted, ENSEMBLE_KWARGS['saved_ensemble_size'])
    pred = fitted.predict_many(X=X)
    assert len(pred) == len(fitted.ensemble)


@dist_test
def test_supervised_feat_select_sampler(client=None):
    '''Has a ModifySample step to get necessary y data'''

    pipe = Pipeline([steps.Flatten(),
                steps.ModifySample(example_get_y),
                steps.SelectPercentile(score_func=f_classif, percentile=50),
                SGDClassifier()])
    en = dict(method_kwargs=dict(classes=[0, 1, 2]), **ENSEMBLE_KWARGS)
    en.update(SAMPLER_DATA_SOURCE)
    fitted = pipe.fit_ensemble(**en)
    ens = fitted.ensemble
    _train_asserts(fitted, en['saved_ensemble_size'])
    pred = fitted.predict_many(**SAMPLER_DATA_SOURCE)
    assert len(pred) == len(SAMPLER_DATA_SOURCE['args_list']) * len(ens)


@dist_test
def test_supervised_feat_select_X_y(client=None):
    '''Has a ModifySample step to get necessary y data'''
    pipe = Pipeline([steps.Flatten(),
            steps.SelectPercentile(score_func=f_classif, percentile=50),
            SGDClassifier()])
    en = dict(method_kwargs=dict(classes=[0, 1, 2]), **ENSEMBLE_KWARGS)
    en.update(X_Y_DATA_SOURCE)
    fitted = pipe.fit_ensemble(**en)
    _train_asserts(fitted, en['saved_ensemble_size'])
    pred = fitted.predict_many(**X_Y_DATA_SOURCE)
    assert len(pred) == len(fitted.ensemble)


@dist_test
def test_kmeans_model_selection(client=None):

    pipe = Pipeline([steps.Flatten(),
                     ('pca', steps.Transform(IncrementalPCA())),
                    ('kmeans', MiniBatchKMeans(n_clusters=5))],
                    scoring=kmeans_aic,
                    scoring_kwargs={'score_weights': [-1]})

    def samp(*args, **kwargs):
        return random_elm_store(bands=12,
                mn=0, mx=1, height=20, width=40)
    en = ENSEMBLE_KWARGS.copy()
    n_clusters_choices = list(range(3, 10))
    def init(pipe, **kwargs):
        estimators = []
        for _ in range(100):
            n_components = np.random.choice(np.arange(2, 6))
            n_clusters = np.random.choice(n_clusters_choices)
            estimator = copy.deepcopy(pipe)
            estimator.set_params(kmeans__n_clusters=n_clusters,
                                 pca__n_components=n_components)
            estimators.append(estimator)
        return estimators
    en['ngen'] = 20
    en['model_scoring'] = kmeans_aic
    en['ensemble_init_func'] = init
    en['model_selection_kwargs'] = dict(drop_n=30, evolve_n=30,
                                        choices=n_clusters_choices)
    en['model_selection'] = kmeans_model_averaging
    sa = SAMPLER_DATA_SOURCE.copy()
    sa['sampler']  = samp
    en.update(sa)
    fitted = pipe.fit_ensemble(**en)
    assert len(fitted.ensemble) == en['saved_ensemble_size']
    preds = fitted.predict_many(**sa)
    assert len(preds) == len(fitted.ensemble) * len(SAMPLER_DATA_SOURCE['args_list'])
