from __future__ import absolute_import, division, print_function, unicode_literals

from itertools import product

from elm.pipeline import steps, Pipeline
from elm.tests.util import (catch_warnings, make_X_y, TESTED,
                            TRANSFORMERS, ALL_STEPS, SLOW,
                            SKIP, REQUIRES_1D, get_params_for_est,
                            PREPROC, pipeline_skip)
from xarray_filters import MLDataset
from xarray_filters.pipeline import Generic

import numpy as np
import pytest


def skipif(modul, name):
    if modul in SKIP or name in SKIP:
        pytest.skip('elm.pipeline.{}.{} - not implemented'.format(modul, name))

@catch_warnings
@pytest.mark.parametrize('modul, name', TESTED.keys())
def test_fit_predict_estimator(modul, name):
    try:
        estimator = TESTED[(modul, name)]
        skipif(modul, name)
        X, y, params, kw = get_params_for_est(estimator, name)
        mod = estimator(**params)
        fitted = mod.fit(X, y)
        assert isinstance(fitted, estimator)
        pred = fitted.predict(X)
        assert isinstance(pred, MLDataset)
        assert tuple(pred.data_vars) == ('predict',)
        # TODO - this should work assert tuple(pred.predict.dims) == tuple(X.dims)
    except:
        print(modul, name, estimator)
        raise


def new_pipeline(*args, flatten_first=True):
    trans = []
    for idx, model in enumerate(args):
        parts = model._cls.__name__.split('.')
        name = parts[-1]
        if any(part in SKIP for part in parts):
            pytest.skip('{} - not implemented'.format(model._cls.__name__))
        out = get_params_for_est(model, name)
        if idx == 0:
            X, y, params, data_kw = out
        else:
            _, _, params, data_kw = out
        if 'score_func' in params:
            if y is None:
                val = X.to_features().features.values
                y = val.dot(np.random.uniform(0, 1, val.shape[1]))
        trans.append(('step_{}'.format(idx + 1), model(**params)))
        if data_kw['is_classifier']:
            y = (y > y.mean()).astype(np.int32)

    if flatten_first:
        def to_feat(X, y=None):
            if hasattr(X, 'to_features'):
                return X.to_features()
            return X
        flatten = Generic(func=to_feat)
        trans = [('step_0', flatten)] + trans
    pipe = Pipeline(trans)
    return pipe, X, y

pipe_combos = product(TRANSFORMERS.keys(), TESTED.keys())
modules_names = [(k1, v1, k2, v2)
                 for (k1, v1), (k2, v2) in pipe_combos]
modules_names_marked = [(item if not any(s in item for s in SLOW) else pytest.mark.slow(item))
                        for item in modules_names
                        if not item[1] in PREPROC]


@catch_warnings
@pytest.mark.parametrize('m1, n1, m2, n2', modules_names_marked)
def test_pipeline_combos(m1, n1, m2, n2):
    pipeline_skip(m1, n1, m2, n2)
    transformer = TRANSFORMERS[(m1, n1)]
    estimator = TESTED[(m2, n2)]
    pipe, X, y = new_pipeline(transformer, estimator)
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert isinstance(pred, MLDataset)



