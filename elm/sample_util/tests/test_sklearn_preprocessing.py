import copy
import os

import numpy as np
import pytest
from sklearn.decomposition import PCA
import yaml

from elm.config import DEFAULTS, ConfigParser
from elm.pipeline.tests.util import (random_elm_store,
                                     tmp_dirs_context,
                                     test_one_config as tst_one_config,
                                     BANDS)
from elm.readers import *
from elm.sample_util.sample_pipeline import create_sample_from_data_source, run_pipeline
from elm.readers.tests.util import (ELM_HAS_EXAMPLES,
                                    ELM_EXAMPLE_DATA_PATH)
transform_model = [('tag_0', PCA(n_components=3))]

@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
               reason='elm-data repo has not been cloned')
def tst_one_pipeline(pipeline,
                            es,
                            run_it=False,
                            tag='tests_of_pipeline'):
    def write(content):
        with open(tag + '.yaml', 'w') as f:
            f.write(content)

    config = copy.deepcopy(DEFAULTS)

    step1 = config['pipeline'][0]
    train_or_predict_dict = copy.deepcopy(config['train']['kmeans'])
    data_source = config['data_sources'][step1['data_source']]
    data_source['get_y_func'] = 'elm.pipeline.tests.util:example_get_y_func_binary'
    data_source['top_dir'] = ELM_EXAMPLE_DATA_PATH
    if not run_it:
        config = ConfigParser(config=config)
        step = config.pipeline[0]['steps'][0]
        for step1 in config.pipeline:
            step1['pipeline'] = pipeline
            pipe = create_sample_from_data_source(pipeline, config, step,
                                    data_source)
            sample, sample_y, sample_weight = run_pipeline(pipe, sample=es,
                                         transform_model=transform_model)

            return sample
    else:
        transform_name = config['pipeline'][0]['steps'][0]['transform']
        for item in config['pipeline']:
            item['pipeline'] = pipeline
        with tmp_dirs_context(tag) as (train_path, predict_path, transform_path, cwd):
            out = tst_one_config(config=config, cwd=cwd)
            assert len(os.listdir(train_path))
            assert len(os.listdir(transform_path))
            assert len(os.listdir(predict_path))

def test_func_scaler():
    es = flatten(random_elm_store(BANDS))
    es2 = es.copy()
    values = es.flat.values.copy()
    values[values <= 0] = 0.0001
    sp = [{'sklearn_preprocessing': 'log10'},{'flatten': 'C'}]
    log10_changed = tst_one_pipeline(sp, es, tag='test_func_scaler')
    assert np.all(log10_changed.flat.values == np.log10(values))
    sp2 = [{'flatten': 'C'},
           {'sklearn_preprocessing': 'require_positive'},
           {'sklearn_preprocessing': 'log10'},
           ]
    log10_changed2 = tst_one_pipeline(sp2, es2, tag='test_func_scaler2')
    assert np.all(log10_changed2.flat.values == log10_changed.flat.values)


def test_standard_scaler():
    es = flatten(random_elm_store(BANDS))
    es.flat.values = np.random.lognormal(100, 1, np.prod(es.flat.shape)).reshape(es.flat.shape)
    sp = [
        {'sklearn_preprocessing': 'require_positive'},
        {'sklearn_preprocessing': 'log10'},
        {'sklearn_preprocessing': 'standard'},
    ]
    scaled = tst_one_pipeline(sp, es, tag='test_standard_scaler')
    mean = np.mean(scaled.flat.values)
    assert mean < 0.1 and mean > -0.1
    std = np.std(scaled.flat.values)
    assert std > 0.9 and std < 1.1


@pytest.mark.slow
def test_scaling_full_config():
    es = random_elm_store(BANDS)
    sp =  [
          {'flatten': 'C'},
          {'get_y': True},
    ]
    sp += DEFAULTS['pipelines']['standardize_log10_var_top_80_inter']
    tst_one_pipeline(sp, es,
                            run_it=True,
                            tag='standardize_log10_var_top_80_inter')


