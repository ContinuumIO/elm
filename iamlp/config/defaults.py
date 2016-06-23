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
                ('train', dict),
                ('predict', dict),
                ('pipeline', list),
            ]
for key in CONFIG_KEYS:
    globals()['DEFAULT_{}'.format(key[0].upper())] = DEFAULTS[key[0]]
del key