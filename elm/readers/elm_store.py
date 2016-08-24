'''

'''


from collections import (namedtuple,
                         Sequence,
                         OrderedDict)
import copy
from functools import wraps
import gc

from matplotlib import pyplot as plt
import gdal
import numpy as np
import xarray as xr

from elm.readers.util import canvas_to_coords, add_es_meta

__all__ = ['ElmStore',
           'canvas_select_elm_store',
           'drop_na_rows',
           'flatten',
           'filled_flattened',
           'check_is_flat',
           'inverse_flatten',
           'data_arrays_as_columns',
           'columns_as_data_arrays',
           ]


class ElmStore(xr.Dataset):
    # not sure if we want a special variant of the xr.Dataset class or not!
    def __init__(self, *args, **kwargs):
        super(ElmStore, self).__init__(*args, **kwargs)
        add_es_meta(self)
        for band in self.data_vars:
            pass
            #print(getattr(self, band).attrs.keys())
            #assert 'canvas' in self.attrs
            #assert 'canvas' in getattr(self, band).attrs
            #print('assert ok')

    def select_canvas(self, canvas):
        return canvas_select_elm_store(self, canvas)

    def drop_na_rows(self):
        assert self.is_flat
        return drop_na_rows(self)

    def flatten(self, ravel_order='F'):
        return flatten(self, ravel_order=ravel_order)

    def filled_flattened(self):
        assert self.is_flat
        return filled_flattened(self)

    def inverse_flatten(self, **attrs):
        assert self.is_flat
        return inverse_flatten(self, **attrs)

    @property
    def is_flat(self):
        return check_is_flat(self, raise_err=False)

    def __str__(self):
        return "ElmStore:\n" + super().__str__()


def canvas_select_elm_store(es, new_canvas):
    if all(c1 == c2 for c1, c2 in zip(es.canvas, new_canvas)):
        # No selection needed
        return es
    new_coords = canvas_to_coords(new_canvas)
    old_coords = canvas_to_coords(es.canvas)
    old_dims = es.canvas.dims
    new_dims = new_coords.dims
    shp_order = []
    attrs = copy.deepcopy(es.attrs)
    attrs['canvas'] = new_canvas
    for nd in new_dims:
        if not nd in old_dims:
            raise ValueError()
        shp_order.append(old_dims.index(nd))
    new_coords_d = dict(new_coords)
    new_coords_in_order_of_old = [(k, new_coords_d[k]) for k in shp_order]
    es_new_dict = OrderedDict()
    for band in band_get_order:
        band_arr = getattr(es, band)
        band_attr = copy.deepcopy(band_arr.attr)
        band_attr['canvas'] = new_canvas
        old_args = [arr for name, arr in old_coords] + [band_arr.values,]
        interp_args = (arr for name, arr in new_coords_in_order_of_old)
        interp = spi.interpnd(*old_args)
        new_values = interp(*interp_args).transpose(shp_order)
        band_arr_new = xr.DataArray(new_values,
                                    coords=new_coords,
                                    dims=new_dims,
                                    attrs=band_attrs)
        es_new_dict[band] = band_arr_new
    es_new = ElmStore(es_new_dict, attrs=attrs)
    gc.colect()
    return es_new


def drop_na_rows(flat):
    check_is_flat(flat)
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
    store = None
    band_names = [band for idx, band in enumerate(es.band_order)]
    old_canvases = []
    old_dims = []
    if hasattr(es, 'flat') and es.flat.values.ndim == 2:
        return es
    for idx, band in enumerate(band_names):
        band_arr = getattr(es, band, None)
        if band_arr is None:
            raise ValueError(repr(es.data_vars))
        assert hasattr(band_arr, 'canvas')
        old_canvases.append(band_arr.canvas)
        old_dims.append(band_arr.dims)
        if store is None:
            # TODO consider canvas here instead
            # of assume fixed size, but that
            # makes reverse transform harder (is that important?)
            store = np.empty((band_arr.values.size,
                              len(es.data_vars))) * np.NaN
        if band_arr.values.ndim == 1:
            # its already flat
            new_values = band_arr.values
        else:
            new_values = band_arr.values.ravel(order=ravel_order)
        store[:, idx] = new_values
    attrs = {}
    attrs['old_canvases'] = old_canvases
    attrs['old_dims'] = old_dims
    attrs['flatten_data_array'] = True
    attrs.update(copy.deepcopy(es.attrs))
    attrs.update(es.attrs)
    flat = ElmStore({'flat': xr.DataArray(store,
                        coords=[('space', np.arange(store.shape[0])),
                                ('band', band_names)],
                        dims=('space',
                              'band'),
                        attrs=attrs)},
                    attrs=attrs)
    return flat


def filled_flattened(na_dropped):
    check_is_flat(na_dropped)
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
                                     coords=[('space', na_dropped.flat.space),
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
            logger.info(msg)
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
    new_es_dict = OrderedDict()
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
        new_es_dict[band] = data_arr
    return ElmStore(new_es_dict, attrs=attrs)


def data_arrays_as_columns(func):
    '''Decorator to require that an ElmStore is flattened
    to 2-d (bands as columns)'''
    @wraps(func)
    def new_func(es, *args, **kwargs):
        flat = flatten(es)
        na_dropped = drop_na_rows(flat)
        return func(na_dropped, *args, **kwargs)
    return new_func


def columns_as_data_arrays(func):
    @wraps(func)
    def new_func(flat_filled, *args, **kwargs):
        return inverse_flatten(flat_filled)
    return new_func


