import copy
import os

import numpy as np
from sklearn.decomposition import PCA

from elm.config import DEFAULTS, ConfigParser
from elm.pipeline.tests.util import (random_elm_store,
                                     tmp_dirs_context,
                                     test_one_config as tst_one_config,
                                     BANDS)
from elm.sample_util.sample_pipeline import get_sample_pipeline_action_data, run_sample_pipeline

transform_model = [('tag_0', PCA(n_components=3))]
def tst_one_sample_pipeline(sample_pipeline, es, run_it=False, tag='tests_of_sample_pipeline'):
    config = copy.deepcopy(DEFAULTS)
    train_or_predict_dict = copy.deepcopy(config['train']['kmeans'])
    data_source = config['data_sources'][train_or_predict_dict['data_source']]
    data_source['get_y_func'] = 'elm.pipeline.tests.util:example_get_y_func_binary'

    if not run_it:
        for item in config['pipeline']:
            item['sample_pipeline'] += sample_pipeline
        step = config['pipeline'][0]
        config = ConfigParser(config=config)
        transform_name = [s for s in config.pipeline if 'transform' in step][0]['transform']
        action_data = get_sample_pipeline_action_data(train_or_predict_dict,
                                     config,
                                     step)
        sample, sample_y, sample_weight = run_sample_pipeline(action_data, sample=es,
                                     transform_model=transform_model)
        return sample
    else:
        transform_name = config['pipeline'][0]['transform']
        for item in config['pipeline']:
            item['sample_pipeline'] = sample_pipeline
        with tmp_dirs_context(tag) as (train_path, predict_path, transform_path, cwd):
            out = tst_one_config(config=config, cwd=cwd)
            assert len(os.listdir(train_path))
            assert len(os.listdir(transform_path))
            assert len(os.listdir(predict_path))

def test_func_scaler():
    es = random_elm_store(BANDS)
    es2 = es.copy()
    values = es.flat.values.copy()
    values[values <= 0] = 0.0001
    sp = [{'sample_pipeline': 'log10'},{'flatten': True}]
    log10_changed = tst_one_sample_pipeline(sp, es, tag='test_func_scaler')
    assert np.all(log10_changed.flat.values == np.log10(values))
    sp2 = [{'flatten': True},
           {'sklearn_preprocessing': 'require_positive'},
           {'sklearn_preprocessing': 'log10'},
           ]
    log10_changed2 = tst_one_sample_pipeline(sp2, es2, tag='test_func_scaler2')
    assert np.all(log10_changed2.flat.values == log10_changed.flat.values)

def test_standard_scaler_and_interactions():
    es = random_elm_store(BANDS)
    es.flat.values = np.random.lognormal(100, 1, np.prod(es.flat.shape)).reshape(es.flat.shape)
    sp = [{'flatten': True},{'sample_pipeline': 'standardize_log10'}]
    scaled = tst_one_sample_pipeline(sp, es, tag='test_standard_scaler_and_interactions')
    mean = np.mean(scaled.flat.values)
    assert mean < 0.1 and mean > -0.1
    std = np.std(scaled.flat.values)
    assert std > 0.9 and std < 1.1
    sp = [{'flatten': True},{'get_y': True}, {'sample_pipeline': 'standardize_log10_var_top_80_inter'}]
    scaled2 = tst_one_sample_pipeline(sp, es)
    assert scaled2.flat.shape[1] > es.flat.shape[1]


def test_scaling_full_config():
    es = random_elm_store(BANDS)
    sp = [
          {'flatten': True},
          {'get_y': True},
          {'sample_pipeline': 'standardize_log10_var_top_80_inter'},
    ]
    tst_one_sample_pipeline(sp, es,
                            run_it=True,
                            tag='standardize_log10_var_top_80_inter')


