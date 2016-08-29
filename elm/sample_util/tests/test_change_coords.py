import numpy as np

from sklearn.decomposition import PCA
from elm.config import ConfigParser
from elm.pipeline.util import make_model_args_from_config
from elm.pipeline.tests.util import (random_elm_store,
                                     test_one_config as tst_one_config,
                                     tmp_dirs_context)
from elm.readers import *
from elm.sample_util.sample_pipeline import run_sample_pipeline
n_components = 3
data_source = {'sample_from_args_func': random_elm_store,
               'sampler_kwargs': {'attrs': {}}}

train = {'model_init_class': 'sklearn.cluster:MiniBatchKMeans',
         'data_source': 'synthetic',
         'ensemble': 'ex3'}

transform = {'model_init_class': 'sklearn.decomposition:PCA',
             'data_source': 'synthetic',
             'model_init_kwargs': {'n_components': n_components,}}

def make_pipeline(sample_pipeline):
    pipeline = [{'train': 'ex1', 'sample_pipeline': sample_pipeline}]
    return pipeline

def tst_one_sample_pipeline(sample_pipeline):
    sample = random_elm_store()
    config = {'data_sources': {'synthetic': data_source},
              'ensembles': {'ex3': {'saved_ensemble_size': 1}},
              'train': {'ex1': train},
              'transform': {'ex2': transform},
              'pipeline': make_pipeline(sample_pipeline)}
    config = ConfigParser(config=config)
    ma, _ = make_model_args_from_config(config, config.pipeline[0], 'train')
    action_data = ma.fit_args[0]
    transform_model = [('tag_0', PCA(n_components=n_components))]
    new_es, _, _ = run_sample_pipeline(action_data,
                                       sample=sample,
                                       transform_model=transform_model)
    return sample, new_es


def test_flat_and_inverse():
    flat = [{'flatten': 'C'}, {'inverse_flatten': True}, {'transpose': ['y', 'x']}]
    es, new_es = tst_one_sample_pipeline(flat)
    assert np.all(new_es.band_1.values == es.band_1.values)


def test_agg():
    for dim, axis in zip(('x', 'y'), (1, 0)):
        for r in range(2):
            if r == 0:
                agg = [{'agg': {'dim': dim, 'func': 'mean'}}]
            else:
                agg = [{'agg': {'axis': axis, 'func': 'mean'}}]
            es, new_es = tst_one_sample_pipeline(agg)
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
    for name, sample_pipeline in sorted(transpose_examples.items()):
        es, new_es = tst_one_sample_pipeline(sample_pipeline)
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

def modify_coords_example(es, *args, **kwargs):
    for band in es.data_vars:
        band_arr = getattr(es, band)
        band_arr.values /= band_arr.values.mean(axis=0)
    return es


def test_modify_coords():
    modify = [{'modify_coords': 'elm.sample_util.tests.test_change_coords:modify_coords_example'}]
    es, new_es = tst_one_sample_pipeline(modify)
    assert np.all(es.band_1.values == new_es.band_1.values)


def test_agg_inverse_flatten():
    for idx, dims in enumerate((['x', 'y'], ['y', 'x'])):
        for agg_dim in ('x', 'y'):
            agg = {'agg': {'dim': agg_dim, 'func': 'median'}}
            sample_pipeline = [{'transpose': dims},
                               {'flatten': 'C'},
                               {'inverse_flatten': True},
                               {'transpose': dims}]
            es, new_es = tst_one_sample_pipeline(sample_pipeline)
            if idx == 0:
                assert new_es.band_1.shape == es.band_1.values.T.shape
            es, new_es = tst_one_sample_pipeline(sample_pipeline + [agg])
            x1, x2 = (getattr(s.band_1, 'x', None) for s in (es, new_es))
            y1, y2 = (getattr(s.band_1, 'y', None) for s in (es, new_es))
            if agg_dim == 'x':
                assert x1 is not None and x2 is None
                assert y1 is not None and y2 is not None
            else:
                assert y1 is not None and y2 is None
                assert x1 is not None and x2 is not None



