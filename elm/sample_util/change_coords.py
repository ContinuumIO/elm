import numpy as np

from elm.config import ElmConfigError, import_callable
from elm.readers import (select_canvas_elm_store,
                         drop_na_rows as _drop_na_rows,
                         ElmStore,
                         flatten as _flatten,
                         inverse_flatten as _inverse_flatten,
                         Canvas
                       )
'''
select_canvas: example_canvas
flatten: True # to [space, band] dims
drop_na_rows: True
inverse_flatten: ['y', 'x']
change_coords:
modify_sample: {func: "elm.sample_util.util:example_2d_agg", kwargs: {}},
transpose: {new_dims: ['x', 'y']},
agg: {dim: y, func: "numpy:median"} # or axis in place of dim
'''

CHANGE_COORDS_ACTIONS = (
    'select_canvas',
    'flatten',
    'drop_na_rows',
    'inverse_flatten',
    'modify_coords',
    'transpose',
    'agg',
)



def select_canvas(es, key, value, **kwargs):
    band = value
    band_arr = getattr(es, band, None)
    if band_arr is None:
        raise ValueError('Argument to select_canvas should be a band name, e.g. "band_1" (found {})'.format(band))
    new_canvas = band_arr.canvas
    new_es = select_canvas_elm_store(es, new_canvas)
    return new_es


def flatten(es, key, value, **kwargs):
    if value != 'C':
        raise ElmConfigError('flatten order argument {} != "C"')
    flat_es = _flatten(es, ravel_order=value)
    return flat_es


def drop_na_rows(es, key, value, **kwargs):
    if not es.is_flat():
        raise ElmConfigError('"flatten" must be called before "drop_na_rows"')
    return _drop_na_rows(es)


def inverse_flatten(es, key, value, **kwargs):
    return _inverse_flatten(es, value)


def modify_coords(es, key, value, **kwargs):
    func = import_callable(value)
    return func(es, **kwargs)


def transpose(es, key, value, **kwargs):
    return es._transpose(value)

def agg(es, key, value, **kwargs):
    return es.agg(**value)


def _check_change_coords_action(config, step, sample_pipeline_step):
    matches = [k for k in sample_pipeline_step if k in CHANGE_COORDS_ACTIONS]
    if not matches or len(matches) > 1:
        raise ElmConfigError('A sample_pipeline step may have exactly 1 key among {}'.format(CHANGE_COORDS_ACTIONS))
    return matches[0]


def change_coords_action(config, step, sample_pipeline_step):
    key = _check_change_coords_action(config, step, sample_pipeline_step)
    value = sample_pipeline_step[key]
    func = 'elm.sample_util.change_coords:{}'.format(key)
    args = (key, value,)
    kwargs = sample_pipeline_step
    return (func, args, kwargs)

