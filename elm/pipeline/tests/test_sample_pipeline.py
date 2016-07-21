import copy

import numpy as np
import pandas as pd
import xarray as xr

from elm.config import DEFAULTS, DEFAULT_TRAIN, ConfigParser
import elm.pipeline.sample_pipeline as sample_pipeline
from elm.preproc.elm_store import ElmStore
from elm.pipeline.tests.util import tmp_dirs_context

BASE = copy.deepcopy(DEFAULTS)

BANDS = ['band_{}'.format(idx + 1) for idx in range(40)]
GEO = (-2223901.039333, 926.6254330549998, 0.0, 8895604.157333, 0.0, -926.6254330549995)

def sample():
    height, width = 100, 80
    val = np.random.uniform(0, 1, width * height * 40).reshape((height * width,len(BANDS)))
    # make half the columns have tiny variance
    val[:, len(BANDS) // 2:] *= 1e-7
    attrs = {'Width': width, 'Height': height, 'GeoTransform': GEO}

    es = ElmStore({'sample': xr.DataArray(val,
                coords=[('space', np.arange(width * height)),
                        ('band', BANDS)],
                dims=['space', 'band'],
                attrs=attrs)},
            attrs=attrs)
    return es

def gen(*args, **kwargs):
    yield from range(5)

BASE['data_source'] = {'sample_args_generator': gen,
                       'sample_from_args_func': sample}


def test_sample_pipeline_feature_selection():
    tag = selection_name = 'variance_selection'
    config = copy.deepcopy(BASE)
    with tmp_dirs_context(tag) as (train_path, predict_path, cwd):
        for idx, action in enumerate(config['pipeline']):
            if 'train' in action or 'predict' in action:
                train_name = action.get('train', action.get('predict'))
                if 'sample_pipeline' in action:
                    action['sample_pipeline'] += [{'feature_selection': selection_name}]
                else:
                    action['sample_pipeline'] = [{'feature_selection': selection_name}]
                config = ConfigParser(config=config)
                config.feature_selection[selection_name] = {
                    'selection': 'sklearn.feature_selection:VarianceThreshold',
                    'threshold': 0.08,
                    'score_func': np.var,
                    'choices': BANDS,
                    'kwargs': {},
                }
                action_data = sample_pipeline.all_sample_ops(config.train[train_name], config, action)

                for repeats in range(100):
                    s = sample_pipeline.run_sample_pipeline(action_data)
                    assert s.sample.shape[1] < 40
                    assert np.mean(s.sample[:len(BANDS) // 2]) > np.mean(s.sample[len(BANDS) // 2:])
                    assert all(b in BANDS for b in s.band)


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
    assert v[np.isnan(v)].size == v2[np.isnan(v2)].size
    v = v[~np.isnan(v)]
    v2 = v2[~np.isnan(v2)]
    assert np.all(v == v2)

