import os
import yaml

from elm.config.util import read_from_egg


YAML_DIR = 'defaults'

DEFAULTS_FILE = os.path.join(YAML_DIR, 'defaults.yaml')
DEFAULTS = read_from_egg(
                DEFAULTS_FILE
                )

CONFIG_KEYS = [('readers',  dict),
                ('sample_args_generators', dict),
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
                ('sample_pipelines', dict),
                ('pipeline', list),
                ('param_grids', dict),
    ]
DEFAULT_TRAIN = tuple(DEFAULTS['train'].values())[0]
DEFAULT_DATA_SOURCE = tuple(DEFAULTS['data_sources'].values())[0]
DEFAULT_READER = tuple(DEFAULTS['readers'].values())[0]
DEFAULT_FEATURE_SELECTOR = tuple(DEFAULTS['feature_selection'].values())[0]

ks = set(globals())
__all__ = [k for k in ks if 'DEFAULT' in k]
__all__ += ['CONFIG_KEYS', 'YAML_DIR']
