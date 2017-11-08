from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import os
import pytest
import shutil
import tempfile
import yaml

from elm.config.config_info import *
from elm.config import ElmConfigError, ConfigParser
from elm.config.tests.fixtures import *

from six import PY2

# "Python 2 + unicode_literals" breaks the YAML parser
if PY2:
    NOT_DICT = (2, 'abc'.encode('utf-8'), 2.2, [2,])
    NOT_INT = ({2:2}, [2, 2], 'abc'.encode('utf-8'), 2.2,)
    NOT_LIST = ({2:2}, 2, 2.2, 'abc'.encode('utf-8'))
    NOT_FUNCTION = 'not_an_importable_function:abc'.encode('utf-8')
else:
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
    pytest.skip('Deprecated (temporarily) elm.config')
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
        bad_config['ensembles'][k]['partial_fit_batches'] = item
        bad_config = tst_bad_config(bad_config)


def test_bad_pipeline():
    pytest.skip('Deprecated (temporarily) elm.config')
    bad_config = copy.deepcopy(DEFAULTS)
    for item in NOT_LIST:
        bad_config['run'] = item
        bad_config = tst_bad_config(bad_config)
        for item in NOT_DICT:
            bad_config['run'][0] = item
            bad_config = tst_bad_config(bad_config)

    # TODO more tests on valid operations
    # e.g. train, predict, resample, etc

