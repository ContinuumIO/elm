from collections import namedtuple

import attr
import numpy as np
import pandas as pd

from elm.sample_util.band_selection import select_from_file
from elm.sample_util.elm_store_concat import elm_store_concat
from elm.sample_util.filename_selection import get_args_list
from elm.sample_util.sample_pipeline import create_sample_from_data_source

_name_num = 0

def _next_name():
    global _name_num
    tag = 'init-sample-{}'.format(_name_num)
    _name_num += 1
    return tag


def _make_sample(pipe, args, sampler, data_source):
    out = pipe.create_sample(X=None, y=None, sample_weight=None,
                             sampler=sampler, sampler_args=args,
                             **data_source)
    return out


def make_samples(pipe, args_list, sampler, data_source):
    dsk = {}
    for arg in args_list:
        sample_name = _next_name()
        dsk[sample_name] = (_make_sample, pipe, arg, sampler, data_source)
    return dsk


def make_samples_dask(X, y, sample_weight, pipe, args_list, sampler, data_source):
    if X is None:
        dsk = make_samples(pipe, args_list, sampler, data_source)
    else:
        dsk = {_next_name(): (lambda: (X, y, sample_weight),)}
    return dsk


def image_selection(filename, **selection_kwargs):

    band_specs = selection_kwargs.get('band_specs', None)
    args_list = selection_kwargs['args_list']
    return select_from_file(filename, band_specs, **selection_kwargs)
