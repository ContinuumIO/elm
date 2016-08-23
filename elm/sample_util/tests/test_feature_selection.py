import copy
from functools import partial

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import chi2

from elm.sample_util.feature_selection import feature_selection_base
from elm.pipeline.tests.util import random_elm_store
# TODO consider moving some default selection like
# these somewhere import-able
SELECTORS = {
    'var': {
        'scoring': None,
        'selection': 'sklearn.feature_selection:VarianceThreshold',
        'kwargs': {'threshold': 0.},
        'choices': "all",
    },
    'kbest_chi2': {
        'scoring': 'chi2',
        'selection': 'sklearn.feature_selection:SelectKBest',
        'kwargs': {'k': 10},
        'choices': "all",
    },
    'kbest_regress': {
        'scoring': 'f_regression',
        'selection': 'sklearn.feature_selection:SelectKBest',
        'kwargs': {'k': 10},
        'choices': "all",
    },
    'kbest_classif': {
        'scoring': 'f_classif',
        'selection': 'sklearn.feature_selection:SelectKBest',
        'kwargs': {'k': 10},
        'choices': "all",
    },
}
for k in set(SELECTORS):
    if 'kbest' in k:
        k2 = k.replace('kbest', 'kpcent')
        SELECTORS[k2] = copy.deepcopy(SELECTORS[k])
        SELECTORS[k2]['kwargs'].pop('k')
        SELECTORS[k2]['kwargs']['percentile'] = 10
        SELECTORS[k2]['selection'] = SELECTORS[k2]['selection'].replace('SelectKBest', 'SelectPercentile')

def sample(ncols):
    bands = ['band_{}'.format(idx + 1) for idx in range(ncols)]

    es = random_elm_store(bands, height=10, width=10)
    nrows = es.flat.values.shape[0]
    samp_y = np.ones(es.flat.space.size)
    samp_y[:samp_y.size // 2] = 0
    return es, samp_y


SAMP, SAMP_Y = sample(100)
TEST_ARGS = tuple((k,v,tf) for k,v in SELECTORS.items() for tf in (True, False))
@pytest.mark.parametrize('name,selection,custom_scorer', TEST_ARGS)
def test_feature_choices_ok(name, selection, custom_scorer):
    samp = SAMP.copy() # so that we don't recreate sample every test
    samp_y = SAMP_Y.copy()
    mult = 4
    keep_columns = []

    selection = copy.deepcopy(selection)
    selection['choices'] = list(samp.sample.band[:samp.sample.shape[1] // mult])
    band_idx = [idx for idx, band in enumerate(samp.sample.band)
                if band in selection['choices']]
    if name == 'var':
        if custom_scorer:
            # already tested
            return
        pcent = 25
        var = np.var(samp.sample.values[:, band_idx],axis=0)
        selection['kwargs']['threshold'] = np.percentile(var, (25,))

    if 'kbest' in name or 'kpcent' in name:
        if custom_scorer:
            def scoring(x, y, **kwargs):
                assert kwargs.get('check_if_kwargs_passed')
                return chi2(x, y)
            selection['scoring'] = scoring
            selection['scoring_kwargs'] = {'check_if_kwargs_passed': True}
        else:
            pass # already has score func in dict from sklearn
    es = feature_selection_base(samp, selection,
                                sample_y=samp_y,
                                keep_columns=keep_columns)
    sel = es.sample.values
    assert not np.any(np.isnan(sel))
    if name == 'var':
        assert sel.shape[1] < len(band_idx)
    if 'kbest' in name:
        assert sel.shape[1] == selection['kwargs']['k'] + len(keep_columns)
    if 'kpcent' in name:
        frac = selection['kwargs']['percentile'] / 100.
        assert abs(sel.shape[1] - frac * (samp.sample.shape[1] / mult) + len(keep_columns)) <= 1
