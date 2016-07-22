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

from elm.config import DEFAULTS, DEFAULT_TRAIN, ConfigParser
import elm.pipeline.sample_pipeline as sample_pipeline
import elm.pipeline.train as elmtrain
import elm.pipeline.predict as predict
from elm.preproc.elm_store import ElmStore
old_ensemble = elmtrain.ensemble
old_predict_step = predict.predict_step

ELAPSED_TIME_FILE = 'elapsed_time_test.txt'

BANDS = ['band_{}'.format(idx + 1) for idx in range(40)]
GEO = (-2223901.039333, 926.6254330549998, 0.0, 8895604.157333, 0.0, -926.6254330549995)

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
        elmtrain.ensemble = return_all
        predict.predict_step = return_all
        yield (elmtrain, predict)
    finally:
        elmtrain.ensemble = old_ensemble
        predict.predict_step = return_all


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

def random_elm_store(bands, mn=0, mx=1, height=100, width=80):
    val = np.random.uniform(mn,
                            mx,
                            width * height * len(bands)).reshape((height * width,len(bands)))
    attrs = {'Width': width,
             'Height': height,
             'GeoTransform': GEO}

    es = ElmStore({'sample': xr.DataArray(val,
                coords=[('space', np.arange(width * height)),
                        ('band', bands)],
                dims=['space', 'band'],
                attrs=attrs)},
            attrs=attrs)
    return es
