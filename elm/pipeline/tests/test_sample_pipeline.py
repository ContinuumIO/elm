import copy

import numpy as np
import pandas as pd
import xarray as xr

from elm.config import DEFAULTS, DEFAULT_TRAIN, ConfigParser
import elm.pipeline.sample_pipeline as sample_pipeline
from elm.preproc.elm_store import ElmStore
from elm.pipeline.tests.util import (tmp_dirs_context,
                                     random_elm_store,
                                     BANDS,
                                     GEO)

BASE = copy.deepcopy(DEFAULTS)



def gen(*args, **kwargs):
    yield from range(5)

def sampler():
    es = random_elm_store(BANDS)
    es.sample.values[:, len(BANDS) // 2:] *= 1e-7
    return es

BASE['data_source'] = {'sample_args_generator': gen,
                       'sample_from_args_func': sampler}


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
                config = ConfigParser(config=BASE)
                config.feature_selection[selection_name] = {
                    'selection': 'sklearn.feature_selection:VarianceThreshold',
                    'scoring': None,
                    'choices': BANDS,
                    'kwargs': {'threshold': 0.08,},
                }
                action_data = sample_pipeline.all_sample_ops(config.train[train_name], config, action)

                for repeats in range(5):
                    s = sample_pipeline.run_sample_pipeline(action_data)
                    assert s.sample.shape[1] < 40
                    assert set(s.sample.band.values) < set(BANDS)


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

