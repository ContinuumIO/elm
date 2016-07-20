from collections import defaultdict
import os

from elm.config.env import parse_env_vars

env = parse_env_vars()

ELM_EXAMPLE_DATA_PATH = env.get('ELM_EXAMPLE_DATA_PATH', None)
if ELM_EXAMPLE_DATA_PATH:
    if not os.path.exists(ELM_EXAMPLE_DATA_PATH):
        raise ValueError('ELM_EXAMPLE_DATA_PATH {} does not exist'.format(ELM_EXAMPLE_DATA_PATH))
    EXAMPLE_DATA_DIRS = [d for d in os.listdir(ELM_EXAMPLE_DATA_PATH) if os.path.isdir(d)]
    EXAMPLE_FILES = defaultdict(lambda: [])
    FILE_TYPES = ('hdf', 'h5', 'tiff', 'nc4') #TODO this may need updates
    for root, dirs, files in os.walk(ELM_EXAMPLE_DATA_PATH):
        files = [os.path.join(root, f) for f in files]
        for f in files:
            parts = f.split('.')
            if len(parts) > 1 and parts[-1] in FILE_TYPES:
                EXAMPLE_FILES[parts[-1]].append(f)
    EXAMPLE_FILES = dict(EXAMPLE_FILES)
else:
    EXAMPLE_FILES = {}