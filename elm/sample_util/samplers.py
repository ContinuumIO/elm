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


def _make_sample(pipe, args, sampler):
    out = pipe.create_sample(X=None, y=None, sample_weight=None,
                              sampler=sampler, sampler_args=args)
    return out


def make_samples(pipe, args_list, sampler):
    dsk = {}
    for arg in args_list:
        sample_name = _next_name()
        dsk[sample_name] = (_make_sample, pipe, arg, sampler)
    return dsk


def make_samples_dask(X, y, sample_weight, pipe, args_list, sampler):
    if X is None:
        dsk = make_samples(pipe, args_list, sampler)
    else:
        dsk = {_next_name(): (lambda: (X, y, sample_weight),)}
    return dsk


def image_selection(band_specs,
                    **selection_kwargs):

    args_list = selection_kwargs['args_list']
    filename = np.random.choice(args_list)
    return select_from_file(filename, band_specs, **selection_kwargs)



def make_one_sample_part(config=None, pipeline=None,
                         data_source=None, transform_model=None,
                         pipe=None, pipeline_kwargs=None,
                         sample=None):
    pipeline_kwargs = pipeline_kwargs or {}
    if pipe is None and sample is None:
        pipe = create_sample_from_data_source(pipeline,
                                                      config=config,
                                                      step=None,
                                                      data_source=data_source,
                                                      **pipeline_kwargs)
    sample, sample_y, sample_weight = run_pipeline(pipe,
                                                          transform_model=transform_model,
                                                          sample=sample)
    return (sample, sample_y, sample_weight)


def make_one_sample(config=None, pipeline=None,
                    data_source=None, transform_model=None,
                    samples_per_batch=None, sample_name=None,
                    pipeline_kwargs=None, sample=None):
    func_args = (make_one_sample_part, config, pipeline,
                 data_source, transform_model, None,
                 pipeline_kwargs, sample)
    dsk = {_next_name(): func_args
           for _ in range(samples_per_batch)}
    dsk.update({sample_name: (elm_store_concat, list(dsk.keys()))})
    return dsk
