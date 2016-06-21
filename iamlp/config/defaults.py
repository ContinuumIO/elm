import os
import yaml

from iamlp.config.util import read_from_egg

YAML_DIR = 'defaults'

DEFAULTS = yaml.safe_load(
            read_from_egg(
                os.path.join(YAML_DIR, 'defaults.yaml')
                )
            )


