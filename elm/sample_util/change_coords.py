import numpy as np

from elm.config import ElmConfigError, import_callable
from elm.readers import (select_canvas as _select_canvas,
                         drop_na_rows as _drop_na_rows,
                         ElmStore,
                         flatten as _flatten,
                         inverse_flatten as _inverse_flatten,
                         Canvas,
                         check_is_flat,
                         transpose as _transpose,
                         aggregate_simple)

CHANGE_COORDS_ACTIONS = (
    'select_canvas',
    'flatten',
    'drop_na_rows',
    'inverse_flatten',
    'modify_sample',
    'transpose',
    'agg',
)


def select_canvas(es, key, value, **kwargs):
    band = value
    band_arr = getattr(es, band, None)
    if band_arr is None:
        raise ValueError('Argument to select_canvas should be a band name, e.g. "band_1" (found {})'.format(band))
    new_canvas = band_arr.canvas
    new_es = _select_canvas(es, new_canvas)
    return new_es


def flatten(es, key, value, **kwargs):
    if value != 'C':
        raise ElmConfigError('flatten order argument {} != "C"')
    flat_es = _flatten(es, ravel_order=value)
    return flat_es


def drop_na_rows(es, key, value, **kwargs):
    if not check_is_flat(es):
        raise ElmConfigError('"flatten" must be called before "drop_na_rows"')
    return _drop_na_rows(es)


def inverse_flatten(es, key, value, **kwargs):
    return _inverse_flatten(es)



def modify_sample(es, key, value, **kwargs):
    func = import_callable(value)
    out = func(es, **kwargs)
    return out


def transpose(es, key, value, **kwargs):
    return _transpose(es, value)


def agg(es, key, value, **kwargs):
    return aggregate_simple(es, **value)


def _check_change_coords_dict_action(pipeline_step):
    matches = [k for k in pipeline_step if k in CHANGE_COORDS_ACTIONS]
    if not matches or len(matches) > 1:
        raise ElmConfigError('A pipeline step may have exactly 1 key among {}'.format(CHANGE_COORDS_ACTIONS))
    return matches[0]


def change_coords_dict_action(pipeline_step):
    if isinstance(pipeline_step, dict):
        key = _check_change_coords_dict_action(pipeline_step)
        value = pipeline_step[key]
        kwargs = pipeline_step
    else:
        key = _check_change_coords_dict_action(pipeline_step._kwargs)
        value = pipeline_step._kwargs[key]
        kwargs = pipeline_step._kwargs
    func = globals()[key]
    args = (key, value,)
    return (func, args, kwargs)

