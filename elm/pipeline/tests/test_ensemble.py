import numpy as np

import pytest
from sklearn.cluster import KMeans

from elm.readers import *
# Below "steps" is a module of all the
# classes which can be used for Pipeline steps
from elm.pipeline import Pipeline, steps
from elm.pipeline.tests.util import random_elm_store

def example_sampler(h, w, bands, **kwargs):
    bands = ['band_{}'.format(idx + 1) for idx in range(bands)]
    return random_elm_store(width=w, height=h, bands=bands)

def test_simple():
    p = Pipeline([steps.Flatten('C'), KMeans(n_clusters=5),])
    args_list = [(100, 200, 5)] * 10 # (height, width, bands)
    data_source = dict(sampler=example_sampler, args_list=args_list)
    ensemble_kw = dict(ngen=2, init_ensemble_size=2)
    tagged_fitted_models = p.ensemble_fit(**data_source, **ensemble_kw)
    (tag1, model1), (tag2, model2) = tagged_fitted_models # ensemble size of 2 here
    X = example_sampler(100, 400, 5)
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)
    assert pred1.shape == pred2.shape == (400 * 100,)
