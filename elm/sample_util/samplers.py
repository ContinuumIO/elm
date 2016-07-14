from collections import namedtuple
import pandas as pd
import numpy as np

from elm.config import delayed, import_callable
from elm.sample_util.filename_selection import get_included_filenames
from elm.sample_util.band_selection import select_from_file

def random_image_selection(band_specs,
                           **selection_kwargs):
    included_filenames = selection_kwargs['included_filenames']
    if not included_filenames:
        raise ValueError('random_image_selection tried to choose from '
                         'included_files but it had no length.\n'
                         'Check "file_generators"')
    filename = np.random.choice(included_filenames)
    return select_from_file(filename, band_specs, **selection_kwargs)

def data_generator(func, config, step, name):
    t = config.train[step['train']]
    s = config.samplers[t['sampler']]
    gen = import_callable(s['data_generator'])
    for sample in gen(**s):
        yield sample
