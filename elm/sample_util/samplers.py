from collections import namedtuple

import attr
import numpy as np
import pandas as pd

from elm.sample_util.band_selection import select_from_file
from elm.sample_util.elm_store_concat import elm_store_concat
from elm.sample_util.filename_selection import get_generated_args
from elm.sample_util.sample_pipeline import (get_sample_pipeline_action_data,
                                             run_sample_pipeline)




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

_name_num = 0

def _next_name():
    global _name_num
    tag = 'load-{}'.format(_name_num)
    _name_num += 1
    return tag


def make_one_sample_part(config=None, sample_pipeline=None,
                         data_source=None, transform_model=None,
                         action_data=None, sample_pipeline_kwargs=None,
                         sample=None):
    sample_pipeline_kwargs = sample_pipeline_kwargs or {}
    if action_data is None and sample is None:
        action_data = get_sample_pipeline_action_data(sample_pipeline,
                                                      config=config,
                                                      step=None,
                                                      data_source=data_source,
                                                      **sample_pipeline_kwargs)
    sample, sample_y, sample_weight = run_sample_pipeline(action_data,
                                                          transform_model=transform_model,
                                                          sample=sample)
    return (sample, sample_y, sample_weight)


def make_one_sample(config=None, sample_pipeline=None,
                    data_source=None, transform_model=None,
                    samples_per_batch=None, sample_name=None,
                    sample_pipeline_kwargs=None, sample=None):
    func_args = (make_one_sample_part, config, sample_pipeline,
                 data_source, transform_model, None,
                 sample_pipeline_kwargs, sample)
    dsk = {_next_name(): func_args
           for _ in range(samples_per_batch)}
    dsk.update({sample_name: (elm_store_concat, list(dsk.keys()))})
    return dsk
