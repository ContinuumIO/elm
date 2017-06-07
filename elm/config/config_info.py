from __future__ import absolute_import, division, print_function, unicode_literals

'''
This module loads elm/config/defaults/config_standard.yaml which
is an elm config used for testing.

TODO: The naming DEFAULTS is misleading because the config is
not used as default settings but rather a testing config.
'''
import os
import yaml

from elm.config.util import read_from_egg


YAML_DIR = 'defaults'

DEFAULTS_FILE = os.path.join(YAML_DIR, 'config_standard.yaml')
DEFAULTS = read_from_egg(
                DEFAULTS_FILE
                )

# elm.config.load_config.ConfigParser
# parses config sections in this order
CONFIG_KEYS = [('readers',  dict),
                ('ensembles', dict),
                ('data_sources', dict),
                ('polys', dict),
                ('resamplers', dict),
                ('aggregations', dict),
                ('masks', dict),
                ('add_features', dict),
                ('feature_selection', dict),
                ('model_scoring', dict),
                ('model_selection', dict),
                ('sklearn_preprocessing', dict),
                ('transform', dict),
                ('train', dict),
                ('predict', dict),
                ('pipelines', dict),
                ('run', list),
                ('param_grids', dict),
    ]
DEFAULT_TRAIN = tuple(DEFAULTS['train'].values())[0]
DEFAULT_DATA_SOURCE = tuple(DEFAULTS['data_sources'].values())[0]
DEFAULT_FEATURE_SELECTOR = tuple(DEFAULTS['feature_selection'].values())[0]

ks = set(globals())
__all__ = [k for k in ks if 'DEFAULT' in k]
__all__ += ['CONFIG_KEYS', 'YAML_DIR']
