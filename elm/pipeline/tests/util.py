from collections import OrderedDict
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
from elm.model_selection.util import filter_kwargs_to_func
import elm.sample_util.sample_pipeline as sample_pipeline
import elm.pipeline.train as elmtrain
import elm.pipeline.predict as predict
import elm.pipeline.transform as elmtransform
from elm.readers import *
from elm.readers.util import add_es_meta
from elm.scripts.main import main as elm_main

old_ensemble = elmtrain.ensemble
old_predict_step = predict.predict_step
old_transform = elmtransform.transform_sample_pipeline_step
old_init_transform = elmtransform.get_new_or_saved_transform_model
ELAPSED_TIME_FILE = 'elapsed_time_test.txt'

BANDS = ['band_{}'.format(idx + 1) for idx in range(40)]
GEO = [-2223901.039333, 926.6254330549998, 0.0, 8895604.157333, 0.0, -926.6254330549995]

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
        elmtransform.transform_sample_pipeline_step = return_all
        elmtransform.get_new_or_saved_transform_model = return_all
        predict.predict_step = return_all

        yield (elmtrain, predict)
    finally:
        elmtrain.ensemble = old_ensemble
        predict.predict_step = old_predict_step
        elmtransform.transform_sample_pipeline_step = old_transform
        elmtransform.get_new_or_saved_transform_model = old_init_transform


@contextlib.contextmanager
def tmp_dirs_context(tag):
    start = datetime.datetime.now()
    tmp1, tmp2, tmp3, tmp4 = (tempfile.mkdtemp() for _ in range(4))
    try:
        old1 = os.environ.get('ELM_TRAIN_PATH') or ''
        old2 =  os.environ.get('ELM_PREDICT_PATH') or ''
        old3 = os.environ.get('ELM_TRANSFORM_PATH') or ''
        os.environ['ELM_TRAIN_PATH'] = tmp1
        os.environ['ELM_PREDICT_PATH'] = tmp2
        os.environ['ELM_TRANSFORM_PATH'] = tmp3
        status = 'ok'
        yield (tmp1, tmp2, tmp3, tmp4)
    except Exception as e:
        status = repr(e)
        raise
    finally:
        os.environ['ELM_TRAIN_PATH'] = old1
        os.environ['ELM_PREDICT_PATH'] = old2
        os.environ['ELM_TRANSFORM_PATH'] = old3
        for tmp in (tmp1, tmp2, tmp3, tmp4):
            if os.path.exists(tmp):
                shutil.rmtree(tmp)
        etime = (datetime.datetime.now() - start).total_seconds()
        with open(ELAPSED_TIME_FILE, 'a') as f:
            f.write('{} {} {} seconds\n'.format(tag, status, etime))



def example_get_y_func_binary(flat_sample):
    '''For use in testing supervised methods which need a get_y_func'''
    col_means = np.mean(flat_sample.flat.values, axis=1)
    med = np.median(col_means)
    ret = np.zeros(col_means.shape, dtype=np.float32)
    ret[col_means > med] = 1
    inds = np.arange(col_means.size)
    np.random.shuffle(inds)
    ret[inds[:3]] = 1
    ret[inds[-3:]] = 0
    return ret


def example_get_y_func_continuous(flat_sample):
    '''For use in testing supervised methods which need a get_y_func'''
    col_means = np.mean(flat_sample.flat.values, axis=1)
    return col_means


def example_custom_continuous_scorer(y_true, y_pred):
    '''This is mean_4th_power_error'''
    return np.mean((y_pred - y_true)**4)


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
    env = copy.deepcopy(os.environ)
    if 'ELM_LOGGING_LEVEL' in env:
        old_val = env['ELM_LOGGING_LEVEL']
    else:
        old_val = None
    env['ELM_LOGGING_LEVEL'] = 'DEBUG'
    sys_argv = ['--config', config_filename, '--echo-config']
    try:
        ret_val = elm_main(sys_argv=sys_argv, return_0_if_ok=False)
    finally:
        if old_val is not None:
            os.environ['ELM_LOGGING_LEVEL'] = old_val
    return ret_val


def random_elm_store(bands=None, mn=0, mx=1, height=100, width=80, **kwargs):
    bands = bands or ['band_{}'.format(idx + 1) for idx in range(3)]
    if isinstance(bands, int):
        bands = ['band_{}'.format(idx + 1) for idx in range(bands)]
    if isinstance(bands[0], (list, tuple)):
        # it is actually band_specs
        bands = [_[-1] for _ in bands]
    get_val = lambda: np.random.uniform(mn,
                            mx,
                            width * height).reshape((height, width))
    attrs = {'width': width,
             'height': height,
             'geo_transform': GEO,
             'canvas': xy_canvas(GEO, width, height, ('y', 'x'))}
    es_dict = OrderedDict()
    for idx, band in enumerate(bands):
        es_dict[band] = xr.DataArray(get_val(),
                                     coords=[('y', np.arange(height)),
                                             ('x', np.arange(width))],
                                     dims=('y', 'x'),
                                     attrs=attrs)
    attrs['band_order'] = bands
    return ElmStore(es_dict, attrs=attrs)


def make_blobs_elm_store(**make_blobs_kwargs):
    '''sklearn.datasets.make_blobs - but return ElmStore
    Parameters:
        as_2d_or_3d:       int - 2 or 3 for num dimensions
        make_blobs_kwargs: kwargs for make_blobs, such as:
                           n_samples=100,
                           n_features=2,
                           centers=3,
                           cluster_std=1.0,
                           center_box=(-10.0, 10.0),
                           shuffle=True,
                           random_state=None'''
    kwargs = filter_kwargs_to_func(make_blobs, **make_blobs_kwargs)
    arr  = make_blobs(**kwargs)[0]
    band = ['band_{}'.format(idx) for idx in range(arr.shape[1])]
    es = ElmStore({'flat': xr.DataArray(arr,
                  coords=[('space', np.arange(arr.shape[0])),
                          ('band', band)],
                  dims=['space', 'band'],
                  attrs={'make_blobs': make_blobs_kwargs})})
    return es


def remove_pipeline_transforms(config):
    config['pipeline'] = [_ for _ in config['pipeline'] if not 'transform' in _]

    for item in config['pipeline']:
        if 'sample_pipeline' in item:
            item['sample_pipeline'] = [_ for _ in item['sample_pipeline'] if not 'transform' in _]

