import copy

import numpy as np
import pandas as pd
import xarray as xr

from elm.config import DEFAULTS, DEFAULT_TRAIN, ConfigParser
from elm.pipeline.tests.util import train_with_synthetic_data
import elm.pipeline.sample_pipeline as sample_pipeline

COLS = ['band_{}'.format(idx + 1) for idx in range(40)]

def sample():
    df = pd.DataFrame(np.random.uniform(0, 1, 25 * 40).reshape(25, 40))
    df.iloc[10] = 0
    df.columns = COLS
    return df


def test_sample_pipeline_feature_selection():
    config, sampler_name, step, idx, train_name = train_with_synthetic_data({}, sample)
    selection_name = tuple(config.feature_selection)[0]
    config.pipeline[idx]['feature_selection'] = step['feature_selection'] = selection_name
    config.feature_selection[selection_name]['selection'] = 'sklearn.feature_selection:VarianceThreshold'
    config.feature_selection[selection_name]['kwargs']['threshold'] = 0.08
    config.train[train_name]['sampler'] = sampler_name
    action_data = sample_pipeline.all_sample_ops(config.train[train_name], config, step)
    for repeats in range(100):
        s = sample_pipeline.run_sample_pipeline(action_data)
        assert s.shape[1] < 40
        assert np.all(s.iloc[10] == 0.)
        assert all(c in COLS for c in s.columns)


def test_elm_store_to_flat_to_elm_store():
    attrs = {'Height': 2,
             'Width': 5,
             'GeoTransform': (-10007554.677, 926.625433055833,
                              0.0, 4447802.078667,
                              0.0, -926.6254330558334)}
    samp_np = np.random.uniform(2,3,4*2*5).reshape(4,2,5)
    samp = xr.Dataset({'sample': xr.DataArray(samp_np,
                    dims=['band', 'y', 'x'], attrs=attrs)}, attrs=attrs)
    flat = sample_pipeline.flatten_cube(samp)
    samp2 = sample_pipeline.flattened_to_cube(flat)
    assert np.all(samp.sample.values == samp2.sample.values)
    assert samp2.attrs.get('dropped_points') == 0
    values = samp.sample.values.copy()
    values[:, 0, 2] = np.NaN
    values[:, 1, 3] = np.NaN
    samp.sample.values = values
    flat_smaller = sample_pipeline.flatten_cube(samp)
    assert flat_smaller.sample.values.shape[0] == samp.sample.values.shape[1] * samp.sample.values.shape[2] - 2
    samp2 = sample_pipeline.flattened_to_cube(flat_smaller)
    v = samp.sample.values
    v2 = samp2.sample.values
    v = v[~np.isnan(v)]
    v2 = v2[~np.isnan(v2)]
    assert np.all(v == v2)

