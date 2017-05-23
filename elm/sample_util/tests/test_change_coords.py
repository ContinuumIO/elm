import numpy as np
import xarray as xr

from sklearn.decomposition import PCA

from elm.config import ConfigParser
from elm.pipeline.tests.util import (test_one_config as tst_one_config,
                                     tmp_dirs_context)
from elm.sample_util.make_blobs import random_elm_store
from earthio.reshape import *
from earthio import ElmStore
from elm.pipeline import Pipeline

X = random_elm_store()

data_source = {'X': X}

train = {'model_init_class': 'sklearn.cluster:MiniBatchKMeans',
         'ensemble': 'ens1'}


def make_run(pipeline, data_source):
    run = [{'data_source': 'synthetic',
            'pipeline': pipeline,
            'train': 'ex1'}]
    return run


def make_config(pipeline, data_source):
    return   {'train': {'ex1': train},
              'data_sources': {'synthetic': data_source},
              'run': make_run(pipeline, data_source),
              'ensembles': {
                'ens1': {
                    'saved_ensemble_size': 1,
                    'init_ensemble_size': 1
                }
              }
            }


def tst_one_pipeline(pipeline,
                     add_na_per_band=0,
                     na_fields_as_str=True,
                     delim='_'):
    from elm.sample_util.sample_pipeline import make_pipeline_steps
    sample = random_elm_store()
    if add_na_per_band:
        for idx, band in enumerate(sample.data_vars):
            band_arr = getattr(sample, band)
            val = band_arr.values
            inds = np.arange(val.size)
            np.random.shuffle(inds)
            x = inds // val.shape[0]
            y = inds % val.shape[0]
            slc = slice(None, add_na_per_band // 2)
            val[y[slc],x[slc]] = 99 * idx
            band_arr.attrs['missing{}value'.format(delim)] = 99 * idx
            slc = slice(add_na_per_band // 2, add_na_per_band)
            val[y[slc], x[slc]] = 199 * idx
            band_arr.attrs['invalid{}range'.format(delim)] = [198 * idx, 200 * idx]
            band_arr.attrs['valid{}range'.format(delim)] = [-1e12, 1e12]
            if na_fields_as_str:
                for field in ('missing{}value', 'invalid{}range', 'valid{}range'):
                    field = field.format(delim)
                    v = band_arr.attrs[field]
                    if isinstance(v, list):
                        band_arr.attrs[field] = ', '.join(map(str,v))
                    else:
                        band_arr.attrs[field] = str(v)
            assert val[np.isnan(val)].size == 0
    config = ConfigParser(config=make_config(pipeline, data_source))
    pipe = Pipeline(make_pipeline_steps(config, pipeline))
    new_es = pipe.fit_transform(sample)
    return sample, new_es[0]


def test_flat_and_inverse():

    flat = [{'flatten': 'C'}, {'inverse_flatten': True}, {'transpose': ['y', 'x']}]
    es, new_es = tst_one_pipeline(flat)
    assert np.all(new_es.band_1.values == es.band_1.values)


def test_agg():
    for dim, axis in zip(('x', 'y'), (1, 0)):
        for r in range(2):
            if r == 0:
                agg = [{'agg': {'dim': dim, 'func': 'mean'}}]
            else:
                agg = [{'agg': {'axis': axis, 'func': 'mean'}}]
            es, new_es = tst_one_pipeline(agg)
            assert dim in es.band_1.dims
            assert dim not in new_es.band_1.dims
            means = np.mean(es.band_1.values, axis=axis)
            new_means = new_es.band_1.values
            diff = np.abs(means - new_means)
            assert np.all(diff < 1e-5)


def test_transpose():
    transpose_examples = {
        'xy': [{'transpose': ['x', 'y']}],
        'inv': [{'flatten': 'C'},
         {'transpose': ['band', 'space']},
         {'transpose': ['space', 'band']},
         {'inverse_flatten': True},
         {'transpose': ['y', 'x']},
        ]
    }
    transpose_examples['fl'] = transpose_examples['xy'] + [{'flatten': 'C'}, {'inverse_flatten': True}, ]
    for name, pipeline in sorted(transpose_examples.items()):
        es, new_es = tst_one_pipeline(pipeline)
        if name == 'fl':
            assert es.band_1.values.T.shape == new_es.band_1.values.shape
            assert np.all(es.band_1.values.T == new_es.band_1.values)
        if name == 'xy':
            assert es.band_1.values.shape == (new_es.band_1.values.shape[1], new_es.band_1.values.shape[0])
            assert np.all(es.band_1.values.T == new_es.band_1.values)
        if 'inv' in name:
            assert es.band_1.values.shape == new_es.band_1.values.shape
            diff = es.band_1.values - new_es.band_1.values
            assert np.all(np.abs(diff) < 1e-5)

def modify_sample_example(es, *args, **kwargs):

    new_es = {}
    for band in es.data_vars:
        band_arr = getattr(es, band)
        v = band_arr.values / band_arr.values.mean(axis=0)
        new_es[band] = xr.DataArray(v, coords=band_arr.coords, dims=band_arr.dims)
        v2 = (band_arr.T.values / band_arr.values.mean(axis=1)).T
        new_es[band + '_new'] = xr.DataArray(v2, coords=band_arr.coords, dims=band_arr.dims)
    return ElmStore(new_es, attrs=es.attrs)


def test_modify_sample():
    modify = [{'modify_sample': 'elm.sample_util.tests.test_change_coords:modify_sample_example'}]
    es, new_es = tst_one_pipeline(modify)
    assert np.all([np.all(getattr(es,b).values.shape == getattr(new_es, b).values.shape) for b in es.data_vars])
    new_names = set(es.band_order) - set(new_es.band_order)
    assert all('new' in n for n in new_names)
    flat = flatten(new_es)
    assert not len(set(tuple(flat.flat.band.values)) ^ set(new_es.band_order))
    inv = inverse_flatten(flat)
    for band in inv.data_vars:
        band_arr = getattr(inv, band)
        assert band_arr.values.shape == getattr(new_es, band).values.shape

def test_agg_inverse_flatten():
    for idx, dims in enumerate((['x', 'y'], ['y', 'x'])):
        for agg_dim in ('x', 'y'):
            agg = {'agg': {'dim': agg_dim, 'func': 'median'}}
            pipeline = [{'transpose': dims},
                               {'flatten': 'C'},
                               {'inverse_flatten': True},
                               {'transpose': dims}]
            es, new_es = tst_one_pipeline(pipeline)
            if idx == 0:
                assert new_es.band_1.shape == es.band_1.values.T.shape
            es, new_es = tst_one_pipeline(pipeline + [agg])
            x1, x2 = (getattr(s.band_1, 'x', None) for s in (es, new_es))
            y1, y2 = (getattr(s.band_1, 'y', None) for s in (es, new_es))
            if agg_dim == 'x':
                assert x1 is not None and x2 is None
                assert y1 is not None and y2 is not None
            else:
                assert y1 is not None and y2 is None
                assert x1 is not None and x2 is not None


def test_set_na_from_meta():
    set_na = [{'modify_sample': 'earthio:set_na_from_meta'}]
    for delim in ('_', '-', ' ', '   '):
        for as_str in (True, False):
            es, new_es = tst_one_pipeline(set_na, add_na_per_band=13,
                                          na_fields_as_str=as_str,
                                          delim=delim)
            assert np.all([np.all(getattr(es,b).values.shape == getattr(new_es, b).values.shape) for b in es.data_vars])
            for band in es.data_vars:
                has_nan = getattr(new_es, band).values
                assert has_nan.size - 13 == has_nan[~np.isnan(has_nan)].size
