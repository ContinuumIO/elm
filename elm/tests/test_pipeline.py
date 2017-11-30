from __future__ import absolute_import, division, print_function, unicode_literals

from itertools import product

from elm.pipeline import steps, Pipeline
from elm.tests.util import (catch_warnings, make_X_y, TESTED_ESTIMATORS,
                            TRANSFORMERS, SLOW,
                            SKIP, REQUIRES_1D, get_params_for_est,
                            PREPROC, skip_transformer_estimator_combo)
from xarray_filters import MLDataset
from xarray_filters.pipeline import Generic

import numpy as np
import pytest


def new_pipeline(args, flatten_first=True):
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
        if 'score_func' in params: # Some estimators require "score_func"
                                   # as an argument (and hence y for the
                                   # score_func, even in cases
                                   # where y may not be required by
                                   # other transformers/estimator steps in the
                                   # Pipeline instance)
            if y is None:
                val = X.to_features().features.values
                y = val.dot(np.random.uniform(0, 1, val.shape[1]))
        trans.append(('step_{}'.format(idx + 1), model(**params)))
        if data_kw['is_classifier']:
            y = (y > y.mean()).astype(np.int32)

    if flatten_first:
        # Add a step to convert first from MLDataset with
        # >=1 DataArrays to a single one with a
        # "features" DataArray - see "to_features" in
        # xarray_filters
        def to_feat(X, y=None):
            if hasattr(X, 'to_features'):
                return X.to_features()
            return X, y
        flatten = Generic(func=to_feat)
        trans = [('step_0', flatten)] + trans
    pipe = Pipeline(trans)
    return pipe, X, y


pipe_combos = product(TRANSFORMERS.keys(), TESTED_ESTIMATORS.keys())
modules_names = [(k1, v1, k2, v2)
                 for (k1, v1), (k2, v2) in pipe_combos]
modules_names_marked = [(item if not any(s in item for s in SLOW) else pytest.mark.slow(item))
                        for item in modules_names
                        if not item[1] in PREPROC and
                        not skip_transformer_estimator_combo(*item)]

def tst_pipeline_combos(module1, cls_name1, module2, cls_name2):
    '''Test a combo of steps, e.g. decompostion, PCA, cluster, KMeans
    as arguments.  Assert a Pipeline of those two steps takes
    X as an MLDataset and y as a numpy array'''
    transformer = TRANSFORMERS[(module1, cls_name1)]
    estimator = TESTED_ESTIMATORS[(module2, cls_name2)]
    pipe, X, y = new_pipeline((transformer, estimator))
    pipe.fit(X, y)
    pred = pipe.predict(X)
    #assert isinstance(pred, MLDataset)

@catch_warnings
@pytest.mark.slow # each test is fast but all of them (~2000) are slow together
@pytest.mark.parametrize('module1, cls_name1, module2, cls_name2', modules_names_marked)
def test_all_pipeline_combos(module1, cls_name1, module2, cls_name2):
    tst_pipeline_combos(module1, cls_name1, module2, cls_name2)


subset = sorted((m for m in modules_names_marked if isinstance(m, tuple)), key=lambda x: hash(x))[:80]

@catch_warnings
@pytest.mark.parametrize('module1, cls_name1, module2, cls_name2', subset)
def test_subset_of_pipeline_combos(module1, cls_name1, module2, cls_name2):
    tst_pipeline_combos(module1, cls_name1, module2, cls_name2)




