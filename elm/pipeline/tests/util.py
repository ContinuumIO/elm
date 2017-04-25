
import contextlib
import copy
import datetime
from functools import partial
import os
import shutil
import subprocess as sp
import tempfile
import yaml

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.datasets import make_blobs

from elm.config import DEFAULTS, DEFAULT_TRAIN, ConfigParser
from elm.config.func_signatures import filter_kwargs_to_func
import elm.sample_util.sample_pipeline as pipeline
import elm.pipeline as elm_pipeline
import elm.sample_util.transform as elmtransform
from elm.scripts.main import main as elm_main

old_ensemble = elm_pipeline.ensemble
old_predict = elm_pipeline.predict_many
ELAPSED_TIME_FILE = 'elapsed_time_test.txt'

@contextlib.contextmanager
def patch_ensemble_predict():
    '''This helps test the job of testing
    getting arguments to
    ensemble by changing that function to
    just return its args,kwargs'''
    def return_all(*args, **kwargs):
        '''An empty function to return what is given to it'''
        return args, kwargs
    try:
        elm_pipeline.ensemble = return_all
        elm_pipeline.predict = return_all
        yield (ens, predict)
    finally:
        elm_pipeline.ensemble = old_ensemble
        elm_pipeline.predict_many = old_predict


@contextlib.contextmanager
def tmp_dirs_context(tag):
    start = datetime.datetime.now()
    tmp1, tmp2, tmp3 = (tempfile.mkdtemp() for _ in range(3))
    try:
        old1 = os.environ.get('ELM_TRAIN_PATH') or ''
        old2 =  os.environ.get('ELM_PREDICT_PATH') or ''
        os.environ['ELM_TRAIN_PATH'] = tmp1
        os.environ['ELM_PREDICT_PATH'] = tmp2
        status = 'ok'
        yield (tmp1, tmp2, tmp3)
    except Exception as e:
        status = repr(e)
        raise
    finally:
        os.environ['ELM_TRAIN_PATH'] = old1
        os.environ['ELM_PREDICT_PATH'] = old2
        for tmp in (tmp1, tmp2, tmp3):
            if os.path.exists(tmp):
                shutil.rmtree(tmp)
        etime = (datetime.datetime.now() - start).total_seconds()
        with open(ELAPSED_TIME_FILE, 'a') as f:
            f.write('{} {} {} seconds\n'.format(tag, status, etime))


def example_get_y_func_binary(flat_sample, **kwargs):
    '''For use in testing supervised methods which need a get_y_func'''
    col_means = np.mean(flat_sample.flat.values, axis=1)
    med = np.median(col_means)
    ret = np.zeros(col_means.shape, dtype=np.float32)
    ret[col_means > med] = 1
    inds = np.arange(col_means.size)
    np.random.shuffle(inds)
    ret[inds[:3]] = 1
    ret[inds[-3:]] = 0
    return (flat_sample, ret, kwargs.get('sample_weight'))


def example_get_y_func_continuous(flat_sample, **kwargs):
    '''For use in testing supervised methods which need a get_y_func'''
    col_means = np.mean(flat_sample.flat.values, axis=1)
    inds = np.arange(col_means.size)
    np.random.shuffle(inds)
    col_means[inds[:3]] += np.random.uniform(0, 0.01, 3)
    col_means[inds[-3:]] += np.random.uniform(0, 0.01, 3)
    return col_means


def test_one_config(config=None, cwd=None):

    if not isinstance(config, str):
        config_str = yaml.dump(config or DEFAULTS)
    else:
        config_str = config
    config_filename = os.path.join(cwd, 'config.yaml')
    with open(config_filename, 'w') as f:
        f.write(config_str)
    sys_argv = ['--config', config_filename, '--echo-config', '--elm-logging-level', 'DEBUG']
    return elm_main(sys_argv=sys_argv, return_0_if_ok=False)


