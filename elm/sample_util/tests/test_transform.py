import copy
import glob
import os

import pytest
import yaml

from sklearn.decomposition import IncrementalPCA

from elm.readers import *
from elm.pipeline import steps
from elm.pipeline.tests.util import random_elm_store

X = flatten(random_elm_store())

def _run_assertions(trans, y, sample_weight):
    assert y is None
    assert sample_weight is None
    assert isinstance(trans, ElmStore)
    assert hasattr(trans, 'flat')
    assert tuple(trans.flat.dims) == ('space', 'band')
    assert trans.flat.values.shape[1] == 3
    assert trans.flat.values.shape[0] == X.flat.values.shape[0]


def test_fit_transform():
    t = steps.Transform(IncrementalPCA(n_components=3))
    trans, y, sample_weight = t.fit_transform(X)
    _run_assertions(trans, y, sample_weight)


def test_partial_fit_transform():
    t = steps.Transform(IncrementalPCA(n_components=3), partial_fit_batches=3)
    trans, y, sample_weight = t.fit_transform(X)
    _run_assertions(trans, y, sample_weight)
    t2 = steps.Transform(IncrementalPCA(n_components=3), partial_fit_batches=3)
    with pytest.raises(TypeError):
        t2.partial_fit = None # will try to call this and get TypeError
        t2.fit_transform(X)


def test_fit():
    t = steps.Transform(IncrementalPCA(n_components=3), partial_fit_batches=2)
    fitted = t.fit(X)
    assert isinstance(fitted, steps.Transform)
    assert isinstance(fitted._estimator, IncrementalPCA)
    trans, y, sample_weight = fitted.transform(X)
    _run_assertions(trans, y, sample_weight)

