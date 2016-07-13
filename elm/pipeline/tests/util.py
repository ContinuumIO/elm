import copy
from functools import partial

import numpy as np
import pandas as pd

from elm.config import DEFAULTS, DEFAULT_TRAIN, ConfigParser
import elm.pipeline.sample_util as sample_util

def data_generator_base(sampler_func, **kwargs):
    while True:
        yield sampler_func()


def train_with_synthetic_data(partial_config, sampler_func):
    config = copy.deepcopy(DEFAULTS)
    config.update(partial_config)
    config = ConfigParser(config=config)
    sampler_name = tuple(config.defaults['samplers'])[0]
    step, idx, train_name = [(s,idx, s['train']) for idx, s in enumerate(config.pipeline)
                 if 'train' in s][0]
    config.samplers[sampler_name]['data_generator'] = partial(data_generator_base, sampler_func)
    config.train[train_name]['sampler'] = sampler_name

    return config, sampler_name, step, idx, train_name