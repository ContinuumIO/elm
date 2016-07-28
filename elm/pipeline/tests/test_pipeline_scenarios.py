import copy
import contextlib
import datetime
import os
import tempfile

import pytest
import yaml

from elm.pipeline import pipeline
from elm.config import DEFAULTS, DEFAULTS_FILE, import_callable
from elm.model_selection import ALL_MODELS_DICT
from elm.model_selection import get_args_kwargs_defaults
from elm.pipeline.tests.util import (tmp_dirs_context,
                                     test_one_config as tst_one_config)
from elm.model_selection.sklearn_support import ALL_MODELS_ESTIMATOR_TYPES
config = copy.deepcopy(DEFAULTS)
for step in config['pipeline']:
    if 'train' in step:
        DEFAULT_TRAIN_KEY = step['train']
        DEFAULT_TRAIN = config['train'][step['train']]
    if 'predict' in step:
        DEFAULT_PREDICT_KEY = step['predict']
        DEFAULT_PREDICT = config['train'][step['predict']]

def test_default_config():
    tag = 'run-with-default-config'
    with tmp_dirs_context(tag) as (train_path, predict_path, cwd):
        out = tst_one_config(config=DEFAULTS, cwd=cwd)
        len_train, len_predict = map(os.listdir, (train_path, predict_path))
        assert len_train
        assert len_predict

@contextlib.contextmanager
def new_training_config(**train_kwargs):
    config = copy.deepcopy(DEFAULTS)
    config['train'][DEFAULT_TRAIN_KEY].update(train_kwargs)
    try:
        yield config
    finally:
        config['train'][DEFAULT_TRAIN_KEY] = DEFAULT_TRAIN

def adjust_config_sample_size(config, n_rows):
    '''Add a step to "sample_pipeline" for limiting
    the number of rows to a random subset of n_rows'''
    for step in config['pipeline']:
        if 'train' in step or 'predict' in step:
            random_rows = [{'take_random_rows': n_rows}]
            if 'sample_pipeline' in step:
                step['sample_pipeline'] += random_rows
            else:
                step['sample_pipeline'] = random_rows

# The following slow_models take longer than about 11 seconds
# to fit / predict a sample of size (500, 11) with default init kwargs
slow_models = ('ARDRegression',
               'TheilSenRegressor',
               'GaussianProcess',
               'Birch',
               'LogisticRegressionCV')
# The MultiTask* models are not handled yet
multi_task = ('MultiTaskLasso',
              'MultiTaskElasticNetCV',
              'MultiTaskElasticNet',
              'MultiTaskLassoCV')

def get_type(model_init_class):
    return ALL_MODELS_ESTIMATOR_TYPES[model_init_class]


def tst_sklearn_method(model_init_class, c, n_rows):
    tag = '{}-n_rows-{}'.format(model_init_class, n_rows)
    with tmp_dirs_context(tag) as (train_path, predict_path, cwd):
        # make a small ensemble for simplicity
        default_ensemble =  {
                              'init_ensemble_size': 2,  # how many models to initialize at start
                              'saved_ensemble_size': 1, # how many models to serialize as "best"
                              'n_generations': 1,       # how many model train/select generations
                              'batches_per_gen': 1,     # how many partial_fit calls per train/select generation
                            }
        default_init_kwargs = copy.deepcopy(DEFAULT_TRAIN['model_init_kwargs'])
        # Initialize the model given in arguments
        kwargs = {'model_init_class': model_init_class,
                  'model_selection_func': 'no_selection',
                  'ensemble_kwargs': default_ensemble,
                  'model_init_kwargs': default_init_kwargs}
        if not hasattr(c, 'predict'):
            # TODO: handle models with "fit_transform"
            # or "transform" methods (ones without a "predict")
            pytest.xfail('Has no predict method: not supporting transform methods yet')
        if any(m in model_init_class for m in multi_task):
            pytest.xfail('{} models from sklearn are unsupported'.format(model_init_class))
        method_args, method_kwargs, _ = get_args_kwargs_defaults(c.fit)
        model_scoring = {}
        kwargs['model_init_kwargs'] = {}
        kwargs['model_scoring'] = None
        if any(a.lower() == 'y' for a in method_args):
            #  supervised
            model_type = get_type(model_init_class)
            if model_type == 'classifier':
                kwargs['get_y_func'] = 'elm.pipeline.tests.util:example_get_y_func_binary'
                kwargs['model_scoring'] = 'accuracy_score_cv'
                kwargs['model_selection_func'] = 'elm.model_selection.base:select_top_n_models'
                kwargs['model_selection_kwargs'] = {'top_n': 1}
            else:
                kwargs['get_y_func'] = 'elm.pipeline.tests.util:example_get_y_func_continuous'
                kwargs['model_scoring'] = None
                kwargs['model_selection'] = None
        elif 'MiniBatchKMeans' not in model_init_class:
            kwargs['model_scoring'] = "ensemble_kmeans_scoring"
            kwargs['model_selection_func'] = "elm.model_selection.kmeans:kmeans_model_averaging"
            model_scoring[kwargs['model_scoring']] = {'scoring': 'elm.pipeline.tests.util:example_custom_continuous_scorer'}
        if 'OrthogonalMatchingPursuit' in model_init_class:
            # This is incorrectly classified as a continuous
            # model in the if block a few lines above
            kwargs['model_scoring'] = None
            kwargs['model_selection_func'] = 'no_selection'
        if model_init_class.endswith('CV'):
            kwargs['model_scoring'] = None
            kwargs['model_selection_func'] = None
        if any(s in model_init_class for s in slow_models):
            # These are too slow for most image classification
            # uses
            pytest.skip('{} is too slow for this test'.format(model_init_class))
        methods = set(dir(c))
        if 'partial_fit' in methods:
            kwargs['ensemble_kwargs']['batches_per_gen'] = 2
        else:
            kwargs['ensemble_kwargs']['n_generations'] = 1
        with new_training_config(**kwargs) as config:
            if kwargs['model_scoring']:
                config['model_scoring'][kwargs['model_scoring']].update(model_scoring)
            if n_rows:
                adjust_config_sample_size(config, n_rows)

            log = tst_one_config(config=config, cwd=cwd)
            predict_outputs = os.listdir(os.path.join(predict_path, DEFAULT_TRAIN_KEY))
            train_outputs = os.listdir(os.path.join(train_path, DEFAULT_TRAIN_KEY))
            assert train_outputs
            assert predict_outputs
            pickles = [t for t in train_outputs if t.endswith('.pkl')]
            assert pickles
            netcdfs, xarrays = [[p for p in predict_outputs if p.endswith(end)]
                        for end in ('.nc', '.xr')]
            assert netcdfs
            assert xarrays


@pytest.mark.slow
@pytest.mark.parametrize('model_init_class,func', tuple(ALL_MODELS_DICT.items()))
def test_sklearn_methods_slow(model_init_class, func):
    '''Test running each classifier/regressor/cluster model
    through the default pipeline adjusted as necessary, where
    the training sample size is a full file (None as n_rows)

    pytest.mark.parametrize calls this
    function once for each model_init_class in ALL_MODELS_DICT
    '''
    tst_sklearn_method(model_init_class, func, None)


@pytest.mark.parametrize('model_init_class,func', tuple(ALL_MODELS_DICT.items()))
def test_sklearn_methods_fast(model_init_class, func):
    '''Test running each classifier/regressor/cluster model
    through the default pipeline adjusted as necessary, where
    the training sample size is random small
    subset of one file's rows

    pytest.mark.parametrize calls this
    function once for each model_init_class in ALL_MODELS_DICT
    '''
    tst_sklearn_method(model_init_class, func, 500)

