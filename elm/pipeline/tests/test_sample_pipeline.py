import copy

import numpy as np
import pandas as pd

from elm.config import DEFAULTS, DEFAULT_TRAIN, ConfigParser
import elm.pipeline.sample_util as sample_util
from elm.pipeline.tests.util import train_with_synthetic_data

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
    action_data = sample_util.all_sample_ops(config.train[train_name], config, step)
    for repeats in range(100):
        s = sample_util.run_sample_pipeline(action_data)
        assert s.shape[1] < 40
        assert np.all(s.iloc[10] == 0.)
        assert all(c in COLS for c in s.columns)

