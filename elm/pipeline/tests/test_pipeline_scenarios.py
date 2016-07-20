import copy
import contextlib
from io import StringIO
import os
import shutil
import subprocess as sp
import tempfile

import pytest
import yaml

from elm.pipeline import pipeline
from elm.config import DEFAULTS, DEFAULTS_FILE
from elm.model_selection import (PARTIAL_FIT_MODEL_DICT,
                                 UNSUPERVISED_MODEL_STR)
from elm.model_selection import get_args_kwargs_defaults
config = copy.deepcopy(DEFAULTS)
for step in config['pipeline']:
    if 'train' in step:
        DEFAULT_TRAIN_KEY = step['train']
        DEFAULT_TRAIN = config['train'][step['train']]
    if 'predict' in step:
        DEFAULT_PREDICT_KEY = step['predict']
        DEFAULT_PREDICT = config['train'][step['predict']]


@pytest.yield_fixture
def scenario_fixture():
    tmp1, tmp2, tmp3 = (tempfile.mkdtemp() for _ in range(3))
    old1, old2 = os.environ.get('ELM_PICKLE_PATH') or '', os.environ.get('ELM_PREDICT_PATH') or ''
    os.environ['ELM_PICKLE_PATH'] = tmp1
    os.environ['ELM_PREDICT_PATH'] = tmp2
    yield (tmp1, tmp2, tmp3)
    os.environ['ELM_PICKLE_PATH'] = old1
    os.environ['ELM_PREDICT_PATH'] = old2
    for tmp in (tmp1, tmp2, tmp3):
        if os.path.exists(tmp):
            shutil.rmtree(tmp)


def tst_one_config(pickle_path, predict_path, config=None, cwd=None):

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

def test_default_config(scenario_fixture):
    pickle_path, predict_path, cwd = scenario_fixture
    out = tst_one_config(pickle_path, predict_path, config=DEFAULTS, cwd=cwd)
    len_train, len_predict = map(os.listdir, (pickle_path, predict_path))
    assert len_train
    assert len_predict


class ExpectedFuncCalledError(ValueError):
    pass

@contextlib.contextmanager
def new_training_config(**train_kwargs):
    config = copy.deepcopy(DEFAULTS)
    config['train'][DEFAULT_TRAIN_KEY].update(train_kwargs)
    try:
        yield config
    finally:
        config['train'][DEFAULT_TRAIN_KEY] = DEFAULT_TRAIN_KEY

@pytest.mark.parametrize('model_init_class', UNSUPERVISED_MODEL_STR)
def test_unsupervised_sklearn(scenario_fixture, model_init_class):
    pickle_path, predict_path, cwd = scenario_fixture
    default_ensemble = copy.deepcopy(DEFAULT_TRAIN['ensemble_kwargs'])
    kwargs = {'model_init_class': model_init_class,
              'model_selection_func': lambda models: model,
              'ensemble_kwargs': copy.deepcopy(default_ensemble)}
    c = import_callable(model_init_class)
    if 'MiniBatchKMeans' not in model_init_class:
        def post_fit_func(model, sample):
            return model
        kwargs['post_fit_func'] = post_fit_func
    methods = set(dir(c))
    if 'partial_fit' in methods:
        kwargs['fit_func'] = 'partial_fit'
        kwargs['ensemble_kwargs']['batches_per_gen'] = 2
    else:
        kwargs['fit_func'] = 'fit'
        kwargs['ensemble_kwargs']['n_generations'] = 1
    with new_training_config(**kwargs) as config:
        log = tst_one_config(pickle_path, predict_path, config=config, cwd=cwd)
        len_train, len_predict = map(os.listdir, (pickle_path, predict_path))
        assert len_train
        assert len_predict

 def pytest_generate_tests(metafunc):
    for model_init_class in UNSUPERVISED_MODEL_STR:
        metafunc.addcall(funcargs=dict(scenario=scenario,
                         subdir=full_folder))
