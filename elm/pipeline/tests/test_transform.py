import copy
import glob
import os

import pytest
import yaml

from elm.config.defaults import DEFAULTS, DEFAULT_TRAIN
from elm.config.util import import_callable
from elm.model_selection.sklearn_support import DECOMP_MODEL_STR
from elm.pipeline.tests.util import (tmp_dirs_context,
                                     test_one_config as tst_one_config)
from elm.model_selection import DECOMP_MODEL_STR

for k,v in DEFAULTS['data_sources'].items():
    DEFAULT_DATA_SOURCE = (k, v)
    if k == 'NPP_DSRF1KD_L2GD':
        break

for k,v in DEFAULTS['transform'].items():
    DEFAULT_TRANSFORM = (k, v)


flatten = {'flatten': 'C'}
def tst_once(tag, config):
    with tmp_dirs_context(tag) as (train_path, predict_path, transform_path, cwd):
        out = tst_one_config(config=config, cwd=cwd)
        assert glob.glob(os.path.join(transform_path, '*','*.pkl'))
        assert not glob.glob(os.path.join(predict_path, '*'))

def tst_transform(model_init_class, is_slow):
    config = copy.deepcopy(DEFAULTS)
    c = import_callable(model_init_class)()
    batch_size = 1000
    params = c.get_params()
    # Make it run faster:
    if 'tol' in params:
        params['tol'] *= 10
    if 'n_components' in params:
        params['n_components'] = 1
    if 'eigen_solver' in params:
        params['eigen_solver'] = 'dense'
    if 'n_topics' in params:
        params['n_topics'] = 2
    if 'batch_size' in params:
        params['batch_size'] = batch_size
    if 'n_jobs' in params:
        params['n_jobs'] = 4
    if 'whiten' in params:
        params['whiten'] = True
    transform = {'pca': {'model_init_kwargs': params,
                            'model_init_class': model_init_class,
                            'data_source': DEFAULT_DATA_SOURCE[0],
                            'ensemble': DEFAULT_TRAIN['ensemble'],
                            }
                }
    if is_slow:
        sp = [flatten, {'transform': 'pca', 'method': 'transform'}]
    else:
        sp = [flatten, {'random_sample': batch_size}]
    pipeline = [{'sample_pipeline': sp,
                 'data_source': DEFAULT_DATA_SOURCE[0],
                 'steps': []}]
    pipeline[0]['steps'] = [{'transform': 'pca',
                             'method':    'fit',}]
    if is_slow:
        pipeline[0]['steps'] += [{'train': 'kmeans'}]

    config['train'] = {'kmeans': copy.deepcopy(DEFAULT_TRAIN)}
    config['pipeline'] = pipeline
    config['data_sources'] = dict([DEFAULT_DATA_SOURCE])
    config['transform'] = transform
    with open('tested_config_{}.yaml'.format(model_init_class.split(':')[-1]), 'w') as f:
        f.write(yaml.dump(config))
    tst_once('test_transform:' + model_init_class, config)


@pytest.mark.parametrize('model_init_class', sorted(DECOMP_MODEL_STR))
def test_transform_pipeline_step(model_init_class):
    tst_transform(model_init_class, False)

