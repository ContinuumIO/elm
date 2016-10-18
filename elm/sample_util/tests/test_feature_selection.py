import numpy as np

from elm.readers import *
from elm.pipeline import steps
from elm.pipeline.tests.util import random_elm_store

X = random_elm_store()
flat_X = flatten(X)
y = flat_X.flat.values.mean(axis=1)
var = np.var(flat_X.flat.values, axis=0)
med = np.median(var)
def test_variance_threshold():

    t = steps.VarianceThreshold(threshold=med, score_func='f_classif')
    X_new, y2, sample_weight = t.fit_transform(flat_X, y)
    assert np.all(y == y2)
    assert sample_weight is None
    assert isinstance(X_new, ElmStore)
    assert hasattr(X_new, 'flat')
    assert X_new.flat.values.shape[1] < flat_X.flat.values.shape[1]


def test_select_percentile():
    t = steps.SelectPercentile(percentile=50, score_func='f_classif')
    X_new, y2, sample_weight = t.fit_transform(flat_X, y)
    assert np.all(y == y2)
    assert sample_weight is None
    assert isinstance(X_new, ElmStore)
    assert hasattr(X_new, 'flat')
    assert X_new.flat.values.shape[1] < flat_X.flat.values.shape[1]


