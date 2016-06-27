import os
import yaml

from iamlp.config.util import read_from_egg

YAML_DIR = 'defaults'

DEFAULTS = read_from_egg(
                os.path.join(YAML_DIR, 'defaults.yaml')
                )

CONFIG_KEYS = [('readers',  dict),
               ('downloads', dict),
                ('data_sources', dict),
                ('file_generators', dict),
                ('file_lists', dict),
                ('samplers', dict),
                ('polys', dict),
                ('resamplers', dict),
                ('aggregations', dict),
                ('masks', dict),
                ('add_features', dict),
                ('feature_selectors', dict),
                ('train', dict),
                ('predict', dict),
                ('pipeline', list),
            ]
DEFAULT_TRAIN = tuple(DEFAULTS['train'].values())[0]
DEFAULT_DATA_SOURCE = tuple(DEFAULTS['data_sources'].values())[0]
DEFAULT_PREDICT = tuple(DEFAULTS['predict'].values())[0]
DEFAULT_SAMPLER = tuple(DEFAULTS['samplers'].values())[0]
DEFAULT_READER = tuple(DEFAULTS['readers'].values())[0]
DEFAULT_FEATURE_SELECTOR = tuple(DEFAULTS['feature_selectors'].values())[0]