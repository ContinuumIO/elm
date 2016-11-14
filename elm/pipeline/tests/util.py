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
import elm.sample_util.sample_pipeline as pipeline
import elm.pipeline as elm_pipeline
import elm.sample_util.transform as elmtransform
from elm.readers import *
from elm.scripts.main import main as elm_main

old_ensemble = elm_pipeline.ensemble
old_predict = elm_pipeline.predict_many
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


def random_elm_store(bands=None, centers=None, std_devs=None, height=100, width=80, **kwargs):
    bands = bands or ['band_{}'.format(idx + 1) for idx in range(3)]
    if isinstance(bands, int):
        bands = ['band_{}'.format(idx + 1) for idx in range(bands)]
    if isinstance(bands[0], (list, tuple)):
        # it is actually band_specs
        bands = [_[-1] for _ in bands]
    if centers is None:
        centers = np.arange(0, len(bands) * 5).reshape((len(bands), 5))
    if std_devs is None:
        std_devs = np.ones((len(bands), 5))
    if len(centers) != len(bands) or len(bands) != len(std_devs):
        raise ValueError('Expected bands, centers, std_devs to have same length')
    if kwargs.get('attrs'):
        attrs = kwargs['attrs']
    else:
        attrs = {'width': width,
                 'height': height,
                 'geo_transform': GEO,
                 'canvas': xy_canvas(GEO, width, height, ('y', 'x'))}
    es_dict = OrderedDict()
    for idx, band in enumerate(bands):
        arr = np.random.normal(centers[idx],
                            std_devs[idx],
                            width * height).reshape((height, width))
        es_dict[band] = xr.DataArray(arr,
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

