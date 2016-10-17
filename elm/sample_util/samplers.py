from collections import namedtuple

import attr
import numpy as np
import pandas as pd

sample_idx = 0
def _next_name(token):
    global sample_idx
    s = '{}_{}'.format(token, sample_idx)
    sample_idx += 1
    return s


def _make_sample(pipe, args, sampler, data_source):
    out = pipe.create_sample(sampler=sampler, sampler_args=args,
                             **{k: v for k, v in data_source.items()
                                if k not in ('sampler', 'sampler_args')})
    return out


def make_samples(pipe, args_list, sampler, data_source):
    dsk = {}
    if not args_list:
        if 'sampler_args' in data_source:
            args_list = [data_source['sampler_args']]
        else:
            raise ValueError('Expected "args_list" or "sampler_args" in data_source')
    for arg in args_list:
        sample_name = _next_name('make_samples_dask')
        dsk[sample_name] = (_make_sample, pipe, arg, sampler, data_source)
    return dsk


def make_samples_dask(X, y, sample_weight, pipe, args_list, sampler, data_source):
    if X is None:
        dsk = make_samples(pipe, args_list, sampler, data_source)
    else:
        dsk = {_next_name('make_samples_dask'): (lambda: (X, y, sample_weight),)}
    return dsk
