import copy

import numpy as np
import pandas as pd

from iamlp.config import DEFAULTS, DEFAULT_TRAIN, ConfigParser
import iamlp.pipeline.sample_util as sample_util

COLS = ['band_{}'.format(idx + 1) for idx in range(40)]
sample = lambda: pd.DataFrame(np.random.uniform(0, 1, 25 * 40).reshape(25, 40))
def data_generator(**kwargs):
    while True:
        df = sample()
        df.iloc[10] = 0
        df.columns = COLS
        yield df

def test_sample_pipeline_feature_selection():
    config = copy.deepcopy(DEFAULTS)
    config = ConfigParser(config=config)
    sampler_name = tuple(config.defaults['samplers'])[0]
    config.samplers[sampler_name]['data_generator'] = data_generator
    step, idx, train_name = [(s,idx, s['train']) for idx, s in enumerate(config.pipeline)
                 if 'train' in s][0]
    selection_name = tuple(config.feature_selection)[0]
    config.pipeline[idx]['feature_selection'] = step['feature_selection'] = selection_name
    config.feature_selection[selection_name]['selection'] = 'sklearn.feature_selection:VarianceThreshold'
    config.feature_selection[selection_name]['kwargs']['threshold'] = 0.08
    config.train[train_name]['sampler'] = sampler_name
    action_data = sample_util.all_sample_ops(config.train[train_name], config, step)
    for repeats in range(100):
        s = sample_util.run_sample_pipeline(action_data)
        assert s.shape[1] < 40
        assert np.all(s.iloc[10] == 0.)
        assert all(c in COLS for c in s.columns)

