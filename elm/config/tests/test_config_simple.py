import copy
import os
import pytest
import shutil
import tempfile
import yaml

from elm.config.defaults import *
from elm.config import ElmConfigError, ConfigParser

NOT_DICT = (2, 'abc', 2.2, [2,])
NOT_INT = ({2:2}, [2, 2], 'abc', 2.2,)
NOT_LIST = ({2:2}, 2, 2.2, 'abc')
NOT_FUNCTION = 'not_an_importable_function:abc'

def dump_config(config):
    tmp = tempfile.mkdtemp()
    config_file = os.path.join(tmp, 'config.yaml')
    with open(config_file, 'w') as f:
        f.write(yaml.dump(config))
    return tmp, config_file

def tst_bad_config(bad_config):
    tmp = None
    try:
        tmp, config_file = dump_config(bad_config)
        with pytest.raises(ElmConfigError):
            ConfigParser(config_file_name=config_file)
    finally:
        if tmp and os.path.exists(tmp):
            shutil.rmtree(tmp)
    # return the ok version for next test
    ok_config = copy.deepcopy(DEFAULTS)
    tmp, config_file = dump_config(ok_config)
    try:
        ConfigParser(config_file) # confirm it is okay for the next test
    finally:
        shutil.rmtree(tmp)
    return ok_config

def test_bad_train_config():

    bad_config = copy.deepcopy(DEFAULTS)
    name = tuple(bad_config['train'].keys())[0]
    for item in NOT_DICT + (None,):
        bad_config['train'][name] = item
        bad_config = tst_bad_config(bad_config)
    for k in bad_config['train'][name]:
        if k.endswith('_func'):
            bad_config['train'][name][k] = NOT_FUNCTION
            bad_config = tst_bad_config(bad_config)
        if k.endswith('_kwargs'):
            for item in NOT_DICT:
                bad_config['train'][name][k] = item
                bad_config = tst_bad_config(bad_config)

    for item in NOT_DICT:
        bad_config['ensembles'] = item
        bad_config = tst_bad_config(bad_config)
    k = tuple(bad_config['ensembles'].keys())[0]
    for item in NOT_INT:
        bad_config['ensembles'][k]['init_ensemble_size'] = item
        bad_config = tst_bad_config(bad_config)
        bad_config['ensembles'][k]['saved_ensemble_size'] = item
        bad_config = tst_bad_config(bad_config)
        bad_config['ensembles'][k]['ngen'] = item
        bad_config = tst_bad_config(bad_config)
        bad_config['ensembles'][k]['batches_per_gen'] = item
        bad_config = tst_bad_config(bad_config)


def test_bad_pipeline():
    bad_config = copy.deepcopy(DEFAULTS)
    for item in NOT_LIST:
        bad_config['pipeline'] = item
        bad_config = tst_bad_config(bad_config)
        for item in NOT_DICT:
            bad_config['pipeline'][0] = item
            bad_config = tst_bad_config(bad_config)

    # TODO more tests on valid operations
    # e.g. train, predict, resample, etc

def test_readers():
    bad_config = copy.deepcopy(DEFAULTS)
    name = tuple(bad_config['readers'].keys())[0]
    for item in NOT_DICT:
        bad_config['readers'] = item
        bad_config = tst_bad_config(bad_config)

        bad_config['readers'][name] = item
        bad_config = tst_bad_config(bad_config)
    bad_config['readers'][name]['load_array'] = NOT_FUNCTION
    bad_config = tst_bad_config(bad_config)
    bad_config['readers'][name]['load_meta'] = NOT_FUNCTION
    bad_config = tst_bad_config(bad_config)


def test_sample_args_generators():
    bad_config = copy.deepcopy(DEFAULTS)
    name = tuple(bad_config['sample_args_generators'].keys())[0]
    for item in NOT_DICT:
        bad_config['sample_args_generators'] = item
        bad_config = tst_bad_config(bad_config)
    bad_config['sample_args_generators'][name] = NOT_FUNCTION
