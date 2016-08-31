from collections import namedtuple
import pandas as pd
import numpy as np

from elm.sample_util.filename_selection import get_generated_args
from elm.sample_util.band_selection import select_from_file
from elm.sample_util.util import InvalidSample

def image_selection(band_specs,
                    **selection_kwargs):
    filename = selection_kwargs.pop('filename', None)
    if not filename:
        # It is assumed all the filenames
        # have been generated

        # and a random choice can be made
        generated_args = selection_kwargs.get('generated_args')
        if not generated_args:
            raise ValueError('image_selection tried to choose randomly from '
                             'generated_args but no args were generated.\n'
                             'Check "sample_args_generators"')
        filename = np.random.choice(generated_args)

    return select_from_file(filename, band_specs, **selection_kwargs)

