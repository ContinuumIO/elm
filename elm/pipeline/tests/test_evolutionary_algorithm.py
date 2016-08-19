import copy
import os

import pytest
import yaml

from elm.config import ConfigParser
from elm.model_selection import MODELS_WITH_PREDICT_DICT
from elm.model_selection.tests.evolve_example_config import CONFIG_STR
from elm.model_selection.util import get_args_kwargs_defaults
from elm.pipeline.tests.util import (tmp_dirs_context,
                                     test_one_config as tst_one_config)

DEFAULT_CONFIG = yaml.load(CONFIG_STR)


def run_one_config(config, tag):
    with tmp_dirs_context(tag) as (train_path, predict_path, transform_path, cwd):
        with open('{}.yaml'.format(tag), 'w') as f:
            f.write(yaml.dump(config))
        out = tst_one_config(config=config, cwd=cwd)
        len_train, len_predict = map(os.listdir, (train_path, predict_path))
        assert os.path.exists(transform_path)
        assert len_train


def test_sklearn_methods_evolutionary():
    '''Same as test_sklearn_methods_fast but uses
    evolutionary_algorithm rather than ensemble
    and only runs the models that have n_clusters
    as an init keyword arg.
    '''
    config = copy.deepcopy(DEFAULT_CONFIG)
    run_one_config(config, 'test_sklearn_methods_evolutionary')
