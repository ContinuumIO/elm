import copy
from itertools import product
import os

import pandas as pd
import pytest
import yaml

from elm.config import ConfigParser
from elm.model_selection import MODELS_WITH_PREDICT_DICT
from elm.model_selection.tests.evolve_example_config import CONFIG_STR
from elm.config.func_signatures import get_args_kwargs_defaults
from elm.pipeline.tests.util import (tmp_dirs_context,
                                     test_one_config as tst_one_config)
from elm.sample_util.make_blobs import make_blobs_elm_store

DEFAULT_CONFIG = yaml.load(CONFIG_STR)

TESTED_CONFIGS = 'tested_configs'
if not os.path.exists(TESTED_CONFIGS):
    os.mkdir(TESTED_CONFIGS)
def run_one_config(config, tag):
    out = None
    with tmp_dirs_context(tag) as (train_path, predict_path, cwd):
        with open(os.path.join(TESTED_CONFIGS, '{}.yaml'.format(tag)), 'w') as f:
            f.write(yaml.dump(config))
        out = tst_one_config(config=config, cwd=cwd)
        len_train, len_predict = map(os.listdir, (train_path, predict_path))
        assert os.path.exists(train_path)
        assert len_train
    return out


def synthetic_centers(n_clusters, n_features):
    base = 2
    return [[base * nf * nc for nf in range(1, n_features + 1)]
            for nc in range(1, 1 + n_clusters * 10000, 10000)]


def tst_finds_true_n_clusters_once(n_clusters, n_features, early_stop):
    pfile = 'kmeans.csv' # EA parameters output CSV from config
    if os.path.exists(pfile):
        os.remove(pfile)
    config = copy.deepcopy(DEFAULT_CONFIG)
    syn = config['data_sources']['synthetic']
    config['train']['kmeans']['model_init_kwargs'] = {
        'n_init': 10,
        'max_iter': 1000,
        'verbose': 0,
    }
    config['train']['kmeans']['model_init_class'] =  'sklearn.cluster:KMeans'
    mb = 'elm.sample_util.make_blobs:random_elm_store'
    syn['sampler'] = mb
    syn.update({'n_samples': 10000,
                'n_features': n_features,
                'centers': synthetic_centers(n_clusters, n_features),
                'std_devs': 0.0000001,})
    syn['sampler_args'] = None
    tag = 'test_sklearn_finds_n_clusters_{}'
    tag = tag.format(n_clusters) + '_' + '_'.join(early_stop.keys() if early_stop else "None")
    pg = config['param_grids']['example_param_grid']
    pg = {k: v for k, v in pg.items() if not k.startswith('pca')}
    pg['control']['mu'] = 16
    pg['control']['k'] = 8
    if not early_stop:
        pg['control']['ngen'] = 3
    else:
        pg['control']['ngen'] = 11
        pg['control']['early_stop'] = early_stop
    pg['kmeans__n_clusters'] = [2, 3, 4, 6,]
    pg['kmeans__init'] = ['k-means++', 'random']
    config['param_grids']['example_param_grid'] = pg
    run_one_config(config, tag)
    assert os.path.exists(pfile)
    params_df = pd.read_csv(pfile)
    pcols = [c for c in params_df.columns
             if not 'objective' in c]
    ocols = [c for c in params_df.columns
             if 'objective' in c]
    assert len(ocols) == 1
    assert len(pcols) == 3
    # the rest asserts no repeated param sets


n_clusters = range(2, 8, 2)
n_features = range(2, 6, 2)
early_stop_conditions = (
    {'abs_change': [1000000000,],'agg': 'all'},
    {'percent_change': [99.99,], 'agg': 'all'},
    {'threshold': [1,],      'agg': 'any'},
    None
)

pytest_args = tuple(product(n_clusters, n_features, early_stop_conditions))
@pytest.mark.flaky(3)
def test_finds_true_num_clusters_fast():
    tst_finds_true_n_clusters_once(*pytest_args[0])


@pytest.mark.slow
@pytest.mark.parametrize('n_clusters, n_features, early_stop', pytest_args)
def test_finds_true_num_clusters_slow(n_clusters, n_features, early_stop):
    tst_finds_true_n_clusters_once(n_clusters, n_features, early_stop)

