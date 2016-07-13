import copy
from functools import partial

import numpy as np
import pandas as pd

from elm.sample_util.feature_selection import feature_selection_base

# TODO consider moving some default selection like
# these somewhere import-able
SELECTORS = {
    'var': {
        'score_func': None,
        'selection': 'sklearn.feature_selection:VarianceThreshold',
        'kwargs': {'threshold': 0.},
        'choices': "all",
    },
    'kbest_chi2': {
        'score_func': 'chi2',
        'selection': 'sklearn.feature_selection:SelectKBest',
        'kwargs': {'k': 10},
        'choices': "all",
    },
    'kbest_regress': {
        'score_func': 'f_regression',
        'selection': 'sklearn.feature_selection:SelectKBest',
        'kwargs': {'k': 10},
        'choices': "all",
    },
    'kbest_classif': {
        'score_func': 'f_classif',
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

def sample(nrows, ncols):
    samp = pd.DataFrame(
            np.random.uniform(0, 1, nrows * ncols).reshape(nrows, ncols)
        )
    samp.columns = ['band_{}'.format(idx + 1) for idx in range(ncols)]
    samp_y = pd.DataFrame(np.ones(samp.shape[0]))
    var = np.var(samp.values, axis=0)
    half_split = np.sort(var)[var.size // 2] + 1e-8
    return samp, samp_y, half_split

def test_feature_selection_ok():
    samp, samp_y, half_split = sample(10000, 20)
    for name, selection in SELECTORS.items():
        selection = copy.deepcopy(selection)
        if name == 'var':
            selection['kwargs']['threshold'] = half_split
        sel = feature_selection_base(samp, selection, sample_y=samp_y)
        if name == 'var':
            high = samp.shape[1] // 2
            low = high - 1 # from taking the sorted midpoint
            assert sel.shape[1] <= high and sel.shape[1] >= low
        if 'kbest' in name:
            assert sel.shape[1] == selection['kwargs']['k']
        if 'kpcent' in name:
            frac = selection['kwargs']['percentile'] / 100.
            assert sel.shape[1] == frac * samp.shape[1]


def test_feature_choices_context_score_func_ok():
    samp_orig, samp_y, half_split = sample(10000, 400)
    extra, _, _ = sample(10000, 5)
    keep_columns = ['extra_info_{}'.format(idx) for idx in range(1, extra.shape[1] + 1)]
    extra.columns = keep_columns
    samp = samp_orig.join(extra)
    mult = 4
    for name, selection in SELECTORS.items():
        selection = copy.deepcopy(selection)
        selection['choices'] = list(samp_orig.columns[:samp_orig.shape[1] // mult])
        if name == 'var':
            selection['kwargs']['threshold'] = 0.005
            low_var_cols = 10
            samp[selection['choices'][:low_var_cols]] *= 1e-7
        if 'kbest' in name or 'kpcent' in name:
            def score_func(x, y, **kwargs):
                out = np.empty((x.shape[1], 2), dtype=np.float64)
                out[:, 0] = np.var(x, axis=0)
                out[:, 1] = np.ones(x.shape[1]) / x.shape[1]
                assert kwargs.get('check_if_kwargs_passed')
                return out[:, 0], out[:, 1]
            selection['score_func'] = score_func
            selection['score_func_kwargs'] = {'check_if_kwargs_passed': True}
        sel = feature_selection_base(samp, selection,
                                    sample_y=samp_y,
                                    keep_columns=keep_columns)
        if name == 'var':
            assert sel.shape[1] == samp_orig.shape[1] // mult - low_var_cols + len(keep_columns)
        if 'kbest' in name:
            assert sel.shape[1] == selection['kwargs']['k'] + len(keep_columns)
        if 'kpcent' in name:
            frac = selection['kwargs']['percentile'] / 100.
            assert sel.shape[1] == frac * (samp_orig.shape[1] / mult) + len(keep_columns)
        assert all(c in selection['choices'] or c in keep_columns for c in sel.columns)
        keep_columns_output = sorted([c for c in sel.columns if c not in selection['choices']])
        assert keep_columns_output == sorted(keep_columns)
