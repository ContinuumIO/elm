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

from elm.config import DEFAULTS, DEFAULT_TRAIN, ConfigParser
import elm.pipeline.sample_pipeline as sample_pipeline

ELAPSED_TIME_FILE = 'elapsed_time_test.txt'

def data_generator_base(sampler_func, **kwargs):
    while True:
        yield sampler_func()

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

def train_with_synthetic_data(partial_config, sampler_func):
    config = copy.deepcopy(DEFAULTS)
    config.update(partial_config)
    config = ConfigParser(config=config)
    sampler_name = tuple(config.defaults['samplers'])[0]
    step, idx, train_name = [(s,idx, s['train']) for idx, s in enumerate(config.pipeline)
                 if 'train' in s][0]
    config.samplers[sampler_name]['data_generator'] = partial(data_generator_base, sampler_func)
    config.train[train_name]['sampler'] = sampler_name
    return config, sampler_name, step, idx, train_name


def example_get_y_func(flat_sample):
    '''For use in testing supervised methods which need a get_y_func'''
    col_means = np.mean(flat_sample.sample.values, axis=1)
    med = np.median(col_means)
    ret = np.zeros(col_means.shape)
    ret[col_means > med] = 1
    return ret

class ExpectedFuncCalledError(ValueError):
    pass

def get_y_func_that_raises(flat_sample):

    raise ExpectedFuncCalledError('From get_y_func')

def get_weight_func_that_raises(flat_sample):

    raise ExpectedFuncCalledError('from get_weight_func')

def test_one_config(config=None, cwd=None):

    config_str = yaml.dump(config or DEFAULTS)
    config_filename = os.path.join(cwd, 'config.yaml')
    with open(config_filename, 'w') as f:
        f.write(config_str)
    proc = sp.Popen(['elm-main',
                      '--config',
                      config_filename,
                      '--echo-config'],
                     cwd=cwd,
                     stdout=sp.PIPE,
                     stderr=sp.STDOUT,
                     env=os.environ)
    r = proc.wait()
    log = proc.stdout.read().decode()
    print(log)
    if r != 0:
        raise ValueError('Error: Bad return code: {}'.format(r))
    assert 'elm.scripts.main - ok' in log
    return log
