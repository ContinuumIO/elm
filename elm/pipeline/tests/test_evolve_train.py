import copy
from itertools import product
import os

import pandas as pd
import pytest
import yaml

from elm.config import ConfigParser
from elm.model_selection import MODELS_WITH_PREDICT_DICT
from elm.model_selection.tests.evolve_example_config import CONFIG_STR
from elm.model_selection.util import get_args_kwargs_defaults
from elm.pipeline.tests.util import (tmp_dirs_context,
                                     test_one_config as tst_one_config,
                                     make_blobs_elm_store)

DEFAULT_CONFIG = yaml.load(CONFIG_STR)


def run_one_config(config, tag):
    out = None
    with tmp_dirs_context(tag) as (train_path, predict_path, transform_path, cwd):
        with open('{}.yaml'.format(tag), 'w') as f:
            f.write(yaml.dump(config))
        out = tst_one_config(config=config, cwd=cwd)
        len_train, len_predict = map(os.listdir, (train_path, predict_path))
        assert os.path.exists(transform_path)
        assert len_train
    return out


def test_sklearn_methods_evolutionary():
    '''Same as test_sklearn_methods_fast but uses
    evolve rather than ensemble
    and only runs the models that have n_clusters
    as an init keyword arg.
    '''
    config = copy.deepcopy(DEFAULT_CONFIG)
    run_one_config(config, 'tested_config_sklearn_methods_EA')


def synthetic_centers(n_clusters, n_features):
    base = 2
    return [[base * nf * nc for nf in range(1, n_features + 1)]
            for nc in range(1, 1 + n_clusters * 10000, 10000)]


def tst_finds_true_n_clusters_once(n_clusters, n_features, early_stop):
    pfile = 'example_param_grid.csv' # EA parameters output CSV from config
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
    mb = 'elm.pipeline.tests.util:make_blobs_elm_store'
    syn['sampler'] = mb
    syn.update({'n_samples': 10000,
                'n_features': n_features,
                'centers': synthetic_centers(n_clusters, n_features),
                'cluster_std': 0.0000001,})
    syn['sampler_args'] = None
    for step in config['pipeline']:
        step['sample_pipeline'] = 'nothing'
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
    pg.pop('sample_pipeline')
    pg.pop('feature_selection')
    config['param_grids']['example_param_grid'] = pg
    ret_val = run_one_config(config, tag)
    kmeans = ret_val['train']['kmeans']
    assert len(kmeans) == 1
    assert len(kmeans[0]) == 2
    kmeans = kmeans[0][1]
    best_n_clusters = kmeans.get_params()['n_clusters']
    assert best_n_clusters == n_clusters
    assert os.path.exists(pfile)
    params_df = pd.read_csv(pfile)
    pcols = [c for c in params_df.columns
             if not 'objective' in c]
    ocols = [c for c in params_df.columns
             if 'objective' in c]
    assert len(ocols) == 1
    assert len(pcols) == 3
    # the rest asserts no repeated param sets
    params = params_df[pcols]
    has_seen = set()
    for idx in range(params.shape[0]):
        row = tuple(params.iloc[idx])
        has_seen.add(row)
    assert len(has_seen) ==  params.shape[0]


n_clusters = range(2, 8, 2)
n_features = range(2, 6, 2)
early_stop_conditions = (
    {'abs_change': [1000000000,],'agg': 'all'},
    {'percent_change': [99.99,], 'agg': 'all'},
    {'threshold': [1,],      'agg': 'any'},
    None
)

pytest_args = tuple(product(n_clusters, n_features, early_stop_conditions))

def test_finds_true_num_clusters_fast():
    tst_finds_true_n_clusters_once(*pytest_args[0])


@pytest.mark.slow
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize('n_clusters, n_features, early_stop', pytest_args)
def test_finds_true_num_clusters_slow(n_clusters, n_features, early_stop):
    tst_finds_true_n_clusters_once(n_clusters, n_features, early_stop)

