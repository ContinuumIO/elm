from collections import (namedtuple,
                         Sequence,
                         OrderedDict)
import copy
from functools import wraps
import gc
import logging
import os

import numpy as np
import scipy.interpolate as spi
import xarray as xr

from elm.config import import_callable
from elm.readers.elm_store import ElmStore
from elm.readers.util import canvas_to_coords

logger = logging.getLogger(__name__)

__all__ = ['select_canvas_elm_store',
           'drop_na_rows',
           'flatten',
           'filled_flattened',
           'check_is_flat',
           'inverse_flatten',
           ]

DEFAULT_INTERPOLATOR = import_callable('scipy.interpolate.interpnd:LinearNDInterpolator')
if os.environ.get('DEFAULT_INTERPOLATOR'):
    DEFAULT_INTERPOLATOR = import_callable(os.environ['DEFAULT_INTERPOLATOR'])


def select_canvas_elm_store(es, new_canvas):
    assert not es.is_flat()
    es_new_dict = OrderedDict()
    for band in es.data_vars:
        data_arr = getattr(es, band)
        if all(c1 == c2 for c1, c2 in zip(data_arr.canvas, new_canvas)):
            new_arr = data_arr
            attrs = data_arr.attrs
        else:
            new_coords = canvas_to_coords(new_canvas)
            old_coords = canvas_to_coords(data_arr.canvas)
            old_dims = data_arr.canvas.dims
            new_dims = new_canvas.dims
            shp_order = []
            attrs = copy.deepcopy(data_arr.attrs)
            attrs['canvas'] = new_canvas
            for nd in new_dims:
                if not nd in old_dims:
                    raise ValueError()
                shp_order.append(old_dims.index(nd))
            index_to_make = xr.Dataset(new_coords)
            data_arr = data_arr.reindex_like(index_to_make, method='nearest')
        es_new_dict[band] = data_arr
    attrs = copy.deepcopy(es.attrs)
    attrs['canvas'] = new_canvas
    es_new = ElmStore(es_new_dict, attrs=attrs)

    return es_new


def drop_na_rows(flat):
    assert flat.is_flat()
    flat_dropped = flat.flat.dropna(dim='space')
    flat_dropped.attrs.update(flat.attrs)
    flat_dropped.attrs['drop_na_rows'] = flat.flat.values.shape[0] - flat_dropped.shape[0]
    attrs = copy.deepcopy(flat.attrs)
    attrs.update(flat_dropped.attrs)
    attrs['shape_before_drop_na_rows'] = flat.flat.values.shape
    no_na = ElmStore({'flat': flat_dropped}, attrs=attrs)
    return no_na


def flatten(es, ravel_order='F'):
    '''Given an ElmStore with dims (band, y, x) return ElmStore
    with shape (space, band) where space is a flattening of x,y

    Params:
        elm_store:  3-d ElmStore (band, y, x)

    Returns:
        elm_store:  2-d ElmStore (space, band)
    '''
    if es.is_flat():
        return es
    if not es.get_shared_canvas():
        raise ValueError('es.select_canvas should be called before flatten when, as in this case, the bands do not all have the same Canvas')
    store = None
    band_names = [band for idx, band in enumerate(es.band_order)]
    old_canvases = []
    old_dims = []
    for idx, band in enumerate(band_names):
        data_arr = getattr(es, band, None)
        if data_arr is None:
            raise ValueError(repr(es.data_vars))
        assert hasattr(data_arr, 'canvas')
        old_canvases.append(data_arr.canvas)
        old_dims.append(data_arr.dims)
        if store is None:
            # TODO consider canvas here instead
            # of assume fixed size, but that
            # makes reverse transform harder (is that important?)
            store = np.empty((data_arr.values.size,
                              len(es.data_vars))) * np.NaN
        if data_arr.values.ndim == 1:
            # its already flat
            new_values = data_arr.values
        else:
            new_values = data_arr.values.ravel(order=ravel_order)
        store[:, idx] = new_values
    attrs = {}
    attrs['canvas'] = es.get_shared_canvas()
    attrs['old_canvases'] = old_canvases
    attrs['old_dims'] = old_dims
    attrs['flatten_data_array'] = True
    attrs.update(copy.deepcopy(es.attrs))
    flat = ElmStore({'flat': xr.DataArray(store,
                        coords=[('space', np.arange(store.shape[0])),
                                ('band', band_names)],
                        dims=('space',
                              'band'),
                        attrs=attrs)},
                    attrs=attrs)
    return flat


def filled_flattened(na_dropped):
    assert na_dropped.is_flat()
    shp = getattr(na_dropped, 'shape_before_drop_na_rows', None)
    if not shp:
        return na_dropped
    filled = np.empty(shp) * np.NaN
    filled[na_dropped.space, :] = na_dropped.flat.values
    attrs = copy.deepcopy(na_dropped.attrs)
    attrs.update(copy.deepcopy(na_dropped.flat.attrs))
    attrs.pop('shape_before_drop_na_rows', None)
    attrs['notnull_shape'] = na_dropped.flat.values.shape
    band = attrs['band_order']
    filled_es = ElmStore({'flat': xr.DataArray(filled,
                                     coords=[('space', np.arange(shp[0])),
                                            ('band', band)],
                                     dims=('space', 'band'),
                                     attrs=attrs)},
                                attrs=attrs)

    return filled_es


def check_is_flat(flat, raise_err=True):
    if not hasattr(flat, 'flat') or not all(hasattr(flat.flat, at) for at in ('space', 'band')):
        msg = 'Expected an ElmStore/Dataset with attribute "flat" and dims ("space", "band")'
        if raise_err:
            raise ValueError(msg)
        else:
            return False
    return True


def inverse_flatten(flat, **attrs):
    '''Given an ElmStore that has been flattened to (space, band) dims,
    return a 3-d ElmStore with dims (band, y, x).  Requires that metadata
    about x,y dims were preserved when the 2-d input ElmStore was created

    Params:
        flat: a 2-d ElmStore (space, band)
        attrs: attribute dict to update the dict of the returned ElmStore

    Returns:
        es:  ElmStore (band, y, x)
    '''
    flat = filled_flattened(flat)
    attrs2 = copy.deepcopy(flat.attrs)
    attrs2.update(copy.deepcopy(attrs))
    attrs = attrs2
    old_canvases = flat.old_canvases
    band_list = list(flat.flat.band_order)
    old_dims = tuple(c.dims for c in old_canvases)
    es_new_dict = OrderedDict()
    attrs['canvas'] = getattr(flat, 'canvas', attrs['canvas'])
    zipped = zip(old_canvases, old_dims, band_list)
    for idx, (old_canvas, dims, band) in enumerate(zipped):
        new_arr = flat.flat.values[:, idx]
        new_coords = canvas_to_coords(old_canvas)
        shp = tuple(new_coords[k].size for k in dims)
        new_arr = new_arr.reshape(shp)
        data_arr = xr.DataArray(new_arr,
                                coords=new_coords,
                                dims=dims,
                                attrs=attrs)
        es_new_dict[band] = data_arr
    return ElmStore(es_new_dict, attrs=attrs)
