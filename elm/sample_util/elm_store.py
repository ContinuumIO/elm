'''
TODO Handle sample_pipeline reshapes like:

data_arrays_as_columns: True
columns_as_data_arrays: True
canvas_select: example_canvas_1
drop_na_rows: True
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




class ElmStore(xr.Dataset):
    # not sure if we want a special variant of the xr.Dataset class or not!

    def __str__(self):
        return "ElmStore:\n" + super().__str__()


def canvas_select_elm_store(es, new_canvas):
    new_coords = canvas_to_coords(new_canvas)
    old_coords = canvas_to_coords(es.canvas)
    old_dims = es.canvas.dims
    new_dims = new_coords.dims
    shp_order = []
    for nd in new_dims:
        if not nd in old_dims:
            raise ValueError()
        shp_order.append(old_dims.index(nd))
    new_coords_d = dict(new_coords)
    new_coords_in_order_of_old = [(k, new_coords_d[k]) for k in shp_order]
    for band in band_get_order:
        band_arr = getattr(es, band)
        old_args = [arr for name, arr in old_coords] + [band_arr.values,]
        interp_args = (arr for name, arr in new_coords_in_order_of_old)
        interp = spi.interpnd(*old_args)
        new_values = interp(*interp_args).transpose(shp_order)
        band_arr_new = xr.DataArray(new_values,
                                    coords=new_coords,
                                    dims=new_dims,
                                    attrs=attrs)


def drop_na_rows(flat):
    flat_dropped = flat.flat.dropna(dim='space')
    flat_dropped.attrs.update(flat.attrs)
    flat_dropped.attrs['drop_na_rows'] = flat.flat.values.shape[0] - flat_dropped.shape[0]
    attrs = copy.deepcopy(flat.attrs)
    attrs.update(flat_dropped.attrs)
    attrs['shape_before_drop_na_rows'] = flat.flat.values.shape
    no_na = ElmStore({'flat': flat_dropped}, attrs=attrs)
    return no_na


def flatten_data_arrays(es, ravel_order='F'):
    '''Given an ElmStore with dims (band, y, x) return ElmStore
    with shape (space, band) where space is a flattening of x,y

    Params:
        elm_store:  3-d ElmStore (band, y, x)

    Returns:
        elm_store:  2-d ElmStore (space, band)
    '''
    store = None
    band_names = [band for idx, band in enumerate(es.BandOrder)]
    old_shapes = []
    old_dims = []
    if all(b == 'flat' for b in es.BandOrder) and len(es.BandOrder) == 1:
        return es
    for idx, band in enumerate(es.BandOrder):
        print(idx, band)
        band_arr = getattr(es, band)
        old_shapes.append(band_arr.values.shape)
        old_dims.append(band_arr.dims)
        if store is None:
            # TODO consider canvas here instead
            # of assume fixed size, but that
            # makes reverse transform harder (is that important?)
            store = np.empty((band_arr.values.size,
                              len(es.data_vars)))
        if band_arr.values.ndim == 1:
            # its already flat
            new_values = band_arr.values
        else:
            new_values = band_arr.values.ravel(order=ravel_order)
        store[:, idx] = new_values
    attrs = {}
    attrs['old_shapes'] = old_shapes
    attrs['old_dims'] = old_dims
    attrs['flatten_data_array'] = True
    attrs.update(copy.deepcopy(es.attrs))
    attrs.update(es.attrs)
    flat = ElmStore({'flat': xr.DataArray(store,
                        coords=[np.arange(store.shape[0]),
                                band_names],
                        dims=('space',
                              'band'),
                        attrs=attrs)},
                    attrs=attrs)
    return flat


def flattened_na_dropped_to_flat_filled(na_dropped):
    shp = na_dropped.shape_before_drop_na_rows
    filled = np.empty(shp) * np.NaN
    filled[na_dropped.space, :] = na_dropped.values
    attrs = copy.deepcopy(na_dropped.attrs)
    attrs.pop('shape_before_drop_na_rows')
    attrs['filled_na_old_shape'] = na_dropped.shape
    filled_es = ElmStore({'flat': xr.DataArray(filled,
                                     coords=[('space', np.arange(np.prod(shp))),
                                            ('band', )],
                                     dims=('space', 'band'),
                                     attrs=attrs)},
                             attrs=attrs)

    return filled_es


def flattened_to_data_arrays(flat, **attrs):
    '''Given an ElmStore that has been flattened to (space, band) dims,
    return a 3-d ElmStore with dims (band, y, x).  Requires that metadata
    about x,y dims were preserved when the 2-d input ElmStore was created

    Params:
        flat: a 2-d ElmStore (space, band)
        attrs: attribute dict to update the dict of the returned ElmStore

    Returns:
        es:  ElmStore (band, y, x)
    '''
    raise NotImplementedError('This needs to be refactored relative to "Canvas"')
    if len(flat.dims) != 2:
        # it's not actually flat
        return flat
    assert hasattr(flat, 'old_shapes')
    assert hasattr(flat, 'band')
    assert hasattr(flat, 'old_dims')
    attrs2 = copy.deepcopy(flat.attrs)
    attrs2.update(copy.deepcopy(attrs))
    attrs = attrs2
    new_es_dict = OrderedDict()
    store = np.empty(filled_na_old_shape)
    for shp, dims, band in zip(flat.old_shapes, flat.old_dims, flat.band):
        filled = getattr(flat, band)
        filled.old_shape
        xr.DataArray(filled, coords)
        space = np.intersect1d(np.arange(np.prod(shp)), flat.space)
    size = attrs['Height'] * attrs['Width']

    row = space // attrs['Width']
    col = space - attrs['Width'] * row
    for band in range(flat.sample.values.shape[1]):
        shp = filled[band, row, col].shape
        reshp = flat.sample.values[:, band].reshape(shp)
        filled[band, row, col] = reshp
    x, y =  row_col_to_xy(np.arange(attrs['Height']),
                  np.arange(attrs['Width']),
                  attrs['GeoTransform'])
    coords = [('band', flat.band), ('y', y), ('x', x)]
    filled = xr.DataArray(filled,
                          coords=coords,
                          dims=['band', 'y', 'x'])
    return ElmStore({'sample': filled}, attrs=attrs)



def data_arrays_as_columns(func):
    '''Decorator to require that an ElmStore is flattened
    to 2-d (bands as columns)'''
    @wraps(func)
    def new_func(es, *args, **kwargs):
        flat = flatten_data_arrays(es)
        na_dropped = drop_na_rows(flat)
        return func(na_dropped, *args, **kwargs)
    return new_func


def columns_as_data_arrays(func):
    @wraps(func)
    def new_func(flat_filled, *args, **kwargs):
        raise NotImplementedError()
        return flattened_to_data_arrays(flat_filled)
    return new_func

