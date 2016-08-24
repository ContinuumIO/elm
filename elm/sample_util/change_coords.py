from elm.config import ElmConfigError
from elm.readers import (canvas_select_elm_store,
                                       drop_na_rows as _drop_na_rows,
                                       ElmStore,
                                       flatten as _flatten,
                                       inverse_flatten as _inverse_flatten,
                                       )
'''
canvas_select: example_canvas
flatten: True # to [space, band] dims
drop_na_rows: True
inverse_flatten: True
change_coords: "elm.sample_util.util:example_2d_agg"
'''


CHANGE_COORDS_ACTIONS = (
    'canvas_select',
    'flatten',
    'drop_na_rows',
    'inverse_flatten',
    'change_coords',
)


OK_DIMS = set(('y', 'x', 'z', 't'))


def canvas_select(es, key, value, **kwargs):
    new_canvas = Canvas(**config.canvases[value])
    new_es = canvas_select_elm_store(es, new_canvas)
    return new_es


def flatten(es, key, value, **kwargs):
    if not value in ('F', 'C'):
        raise ElmConfigError('flatten order argument {} not in ("F", "C", None) - None defaults to "F"')
    flat_es = _flatten(es, ravel_order=value)
    return flat_es


def drop_na_rows(es, key, value, **kwargs):
    if not hasattr(es, 'flat'):
        raise ElmConfigError('"flatten" must be called before "drop_na_rows"')
    return _drop_na_rows(es)


def inverse_flatten(es, key, value, **kwargs):
    return _inverse_flatten(es)


def change_coords(es, key, value, **kwargs):
    func = import_callable(value)
    es = func(es, **kwargs)
    return es

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

